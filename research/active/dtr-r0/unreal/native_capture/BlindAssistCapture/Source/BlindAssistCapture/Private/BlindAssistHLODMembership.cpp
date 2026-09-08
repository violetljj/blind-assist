#include "BlindAssistCaptureLibrary.h"

#include "Engine/Engine.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformMisc.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"
#include "UObject/Package.h"
#include "WorldPartition/HLOD/HLODActor.h"
#include "WorldPartition/HLOD/HLODSourceActorsFromCell.h"
#include "WorldPartition/WorldPartitionRuntimeCell.h"

namespace
{
TArray<TSharedPtr<FJsonValue>> VectorJson(const FVector& Value)
{
    return { MakeShared<FJsonValueNumber>(Value.X), MakeShared<FJsonValueNumber>(Value.Y),
             MakeShared<FJsonValueNumber>(Value.Z) };
}

TSharedRef<FJsonObject> TransformJson(const FTransform& Value)
{
    TSharedRef<FJsonObject> Result = MakeShared<FJsonObject>();
    Result->SetArrayField(TEXT("translation_cm"), VectorJson(Value.GetTranslation()));
    Result->SetArrayField(TEXT("scale"), VectorJson(Value.GetScale3D()));
    const FQuat Rotation = Value.GetRotation();
    Result->SetArrayField(TEXT("rotation_xyzw"), { MakeShared<FJsonValueNumber>(Rotation.X),
        MakeShared<FJsonValueNumber>(Rotation.Y), MakeShared<FJsonValueNumber>(Rotation.Z),
        MakeShared<FJsonValueNumber>(Rotation.W) });
    return Result;
}
}

bool UBlindAssistCaptureLibrary::ExportWorldHLODSourceMembership(UObject* WorldContextObject, const FString& Filename)
{
#if WITH_EDITOR
    UWorld* World = GEngine ? GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::ReturnNull) : nullptr;
    FString CaptureRoot = FPlatformMisc::GetEnvironmentVariable(TEXT("BA_CITY_OUT"));
    FString Output = FPaths::ConvertRelativePathToFull(Filename);
    FPaths::NormalizeFilename(Output);
    if (!World || CaptureRoot.IsEmpty())
    {
        return false;
    }
    CaptureRoot = FPaths::ConvertRelativePathToFull(CaptureRoot);
    FPaths::NormalizeDirectoryName(CaptureRoot);
    if (!FPaths::IsUnderDirectory(Output, CaptureRoot) || IFileManager::Get().FileExists(*Output))
    {
        return false;
    }

    TSharedRef<FJsonObject> Report = MakeShared<FJsonObject>();
    Report->SetStringField(TEXT("schema"), TEXT("city-native-hlod-membership-v1"));
    Report->SetStringField(TEXT("status"), TEXT("UNVERIFIED"));
    Report->SetStringField(TEXT("map_asset"), World->GetOutermost()->GetName());
    Report->SetStringField(TEXT("scope"), TEXT("Loaded HLOD actors only; no source loading, recursive closure, component-instance or renderer visibility admission"));
    TArray<TSharedPtr<FJsonValue>> Proxies;
    int32 Unresolved = 0;
    for (TActorIterator<AWorldPartitionHLOD> It(World); It; ++It)
    {
        const AWorldPartitionHLOD* Actor = *It;
        TSharedRef<FJsonObject> Proxy = MakeShared<FJsonObject>();
        Proxy->SetStringField(TEXT("proxy_actor_path"), Actor->GetPathName());
        Proxy->SetStringField(TEXT("proxy_actor_guid"), Actor->GetActorGuid().ToString(EGuidFormats::DigitsWithHyphens));
        const UPackage* ExternalPackage = Actor->GetExternalPackage();
        Proxy->SetStringField(TEXT("proxy_actor_package"), ExternalPackage ? ExternalPackage->GetName() : Actor->GetOutermost()->GetName());
        const UWorldPartitionHLODSourceActors* Source = Actor->GetSourceActors();
        Proxy->SetStringField(TEXT("source_mapping_class"), Source ? Source->GetClass()->GetPathName() : TEXT("NONE"));
        const UWorldPartitionHLODSourceActorsFromCell* Cell = Cast<UWorldPartitionHLODSourceActorsFromCell>(Source);
        TArray<TSharedPtr<FJsonValue>> Sources;
        bool Complete = Cell != nullptr;
        if (Cell)
        {
            for (const FWorldPartitionRuntimeCellObjectMapping& Mapping : Cell->GetActors())
            {
                TSharedRef<FJsonObject> Item = MakeShared<FJsonObject>();
                Item->SetStringField(TEXT("package"), Mapping.Package.ToString());
                Item->SetStringField(TEXT("path"), Mapping.Path.ToString());
                Item->SetStringField(TEXT("loaded_path"), Mapping.LoadedPath.ToString());
                Item->SetStringField(TEXT("container_package"), Mapping.ContainerPackage.ToString());
                Item->SetStringField(TEXT("world_package"), Mapping.WorldPackage.ToString());
                Item->SetStringField(TEXT("container_id"), Mapping.ContainerID.ToString());
                Item->SetStringField(TEXT("actor_instance_guid"), Mapping.ActorInstanceGuid.ToString(EGuidFormats::DigitsWithHyphens));
                Item->SetStringField(TEXT("base_class"), Mapping.BaseClass.ToString());
                Item->SetStringField(TEXT("native_class"), Mapping.NativeClass.ToString());
                Item->SetBoolField(TEXT("nested_hlod_source"), Mapping.NativeClass == AWorldPartitionHLOD::StaticClass()->GetClassPathName());
                Item->SetObjectField(TEXT("container_transform"), TransformJson(Mapping.ContainerTransform));
                Item->SetObjectField(TEXT("editor_only_parent_transform"), TransformJson(Mapping.EditorOnlyParentTransform));
                const bool Valid = !Mapping.Package.IsNone() && !Mapping.Path.IsNone() && Mapping.ActorInstanceGuid.IsValid();
                Item->SetBoolField(TEXT("source_actor_identity_fields_valid"), Valid);
                Complete &= Valid;
                Sources.Add(MakeShared<FJsonValueObject>(Item));
            }
        }
        Complete &= !Sources.IsEmpty();
        Proxy->SetArrayField(TEXT("sources"), Sources);
        Proxy->SetStringField(TEXT("status"), Complete ? TEXT("SOURCE_ACTOR_MAPPING_EXPORTED_NOT_COMPONENT_INSTANCE_ADMISSION") : TEXT("UNVERIFIED"));
        if (!Complete)
        {
            ++Unresolved;
            Proxy->SetStringField(TEXT("error"), !Cell ? TEXT("Missing or unsupported non-cell source mapping") : TEXT("Empty mapping or invalid source identity fields"));
        }
        Proxies.Add(MakeShared<FJsonValueObject>(Proxy));
    }
    Report->SetNumberField(TEXT("loaded_hlod_count"), Proxies.Num());
    Report->SetNumberField(TEXT("unresolved_hlod_count"), Unresolved);
    Report->SetArrayField(TEXT("hlod"), Proxies);
    FString Text;
    if (!FJsonSerializer::Serialize(Report, TJsonWriterFactory<>::Create(&Text)))
    {
        return false;
    }
    if (!IFileManager::Get().MakeDirectory(*FPaths::GetPath(Output), true))
    {
        return false;
    }
    return FFileHelper::SaveStringToFile(Text, *Output, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM,
        &IFileManager::Get(), FILEWRITE_NoReplaceExisting);
#else
    return false;
#endif
}
