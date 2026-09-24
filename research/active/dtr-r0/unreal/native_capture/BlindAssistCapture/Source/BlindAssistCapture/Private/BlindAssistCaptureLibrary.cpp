#include "BlindAssistCaptureLibrary.h"

#include "AssetCompilingManager.h"
#include "ContentStreaming.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Engine/Engine.h"
#include "Engine/StreamableRenderAsset.h"
#include "Engine/TextureRenderTarget2D.h"
#include "HAL/FileManager.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "Misc/Paths.h"
#include "Serialization/Archive.h"
#include "ShaderCompiler.h"
#include "UObject/UObjectIterator.h"
#include "Materials/Material.h"
#include "Materials/MaterialExpressionConstant.h"
#include "Materials/MaterialExpressionConstant3Vector.h"
#include "Materials/MaterialExpressionSetMaterialAttributes.h"
#include "UObject/Package.h"
#include "Materials/MaterialInterface.h"
#include "Materials/MaterialRelevance.h"
#include "MaterialShared.h"
#include "RHI.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"

DEFINE_LOG_CATEGORY_STATIC(LogBlindAssistCapture, Log, All);

FString UBlindAssistCaptureLibrary::GetIsmPreservationState(UInstancedStaticMeshComponent* Component)
{
    TSharedRef<FJsonObject> Data = MakeShared<FJsonObject>();
    Data->SetStringField(TEXT("schema"), TEXT("cnh_ism_preservation_state_v1"));
    Data->SetStringField(TEXT("data_status"), IsValid(Component) ? TEXT("AVAILABLE") : TEXT("UNAVAILABLE"));
    if (IsValid(Component))
    {
        Data->SetNumberField(TEXT("instance_count"), Component->GetInstanceCount());
        Data->SetNumberField(TEXT("num_custom_data_floats"), Component->NumCustomDataFloats);
        Data->SetNumberField(TEXT("instancing_random_seed"), Component->InstancingRandomSeed);
        TArray<TSharedPtr<FJsonValue>> CustomData;
        CustomData.Reserve(Component->PerInstanceSMCustomData.Num());
        for (float Value : Component->PerInstanceSMCustomData)
        {
            if (!FMath::IsFinite(Value))
            {
                Data->SetStringField(TEXT("data_status"), TEXT("UNAVAILABLE_NONFINITE_CUSTOM_DATA"));
                CustomData.Add(MakeShared<FJsonValueNull>());
            }
            else CustomData.Add(MakeShared<FJsonValueNumber>(static_cast<double>(Value)));
        }
        Data->SetArrayField(TEXT("per_instance_sm_custom_data"), CustomData);
        TArray<TSharedPtr<FJsonValue>> Seeds;
        for (const FInstancedStaticMeshRandomSeed& Seed : Component->AdditionalRandomSeeds)
        {
            TSharedRef<FJsonObject> Entry = MakeShared<FJsonObject>();
            Entry->SetNumberField(TEXT("start_instance_index"), Seed.StartInstanceIndex);
            Entry->SetNumberField(TEXT("random_seed"), Seed.RandomSeed);
            Seeds.Add(MakeShared<FJsonValueObject>(Entry));
        }
        Data->SetArrayField(TEXT("additional_random_seeds"), Seeds);
    }
    FString Result;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Result);
    FJsonSerializer::Serialize(Data, Writer);
    return Result;
}

UMaterialInterface* UBlindAssistCaptureLibrary::GetDefaultSurfaceMaterial()
{
    return UMaterial::GetDefaultMaterial(MD_Surface);
}

FString UBlindAssistCaptureLibrary::GetMaterialGeometryCapability(UMaterialInterface* Material)
{
    TSharedRef<FJsonObject> Data = MakeShared<FJsonObject>();
    Data->SetStringField(TEXT("schema"), TEXT("cnh_material_geometry_capability_v1"));
    Data->SetBoolField(TEXT("scene_admission"), false);
    Data->SetStringField(TEXT("material"), Material ? Material->GetPathName() : TEXT(""));
    Data->SetStringField(TEXT("data_status"), TEXT("UNAVAILABLE"));
    Data->SetStringField(TEXT("capability"), TEXT("UNKNOWN"));
    Data->SetNumberField(TEXT("shader_platform_id"), static_cast<int32>(GMaxRHIShaderPlatform));
    Data->SetNumberField(TEXT("feature_level_id"), static_cast<int32>(GMaxRHIFeatureLevel));
    Data->SetStringField(TEXT("platform_source"), TEXT("ACTIVE_GMaxRHIShaderPlatform"));
    auto Finish = [&Data](const TCHAR* Reason)
    {
        Data->SetStringField(TEXT("reason"), Reason);
        FString Json;
        TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Json);
        FJsonSerializer::Serialize(Data, Writer);
        return Json;
    };
    if (!IsInGameThread()) return Finish(TEXT("GAME_THREAD_REQUIRED"));
    if (!Material) return Finish(TEXT("MATERIAL_MISSING"));
    Data->SetNumberField(TEXT("blend_mode"), static_cast<int32>(Material->GetBlendMode()));
    Data->SetBoolField(TEXT("two_sided"), Material->IsTwoSided());
    Data->SetBoolField(TEXT("is_masked"), Material->IsMasked());
    const EMaterialQualityLevel::Type Quality = GetCurrentMaterialQualityLevelChecked();
    Data->SetNumberField(TEXT("quality_level_id"), static_cast<int32>(Quality));
    FMaterialResource* Resource = Material->GetMaterialResource(GMaxRHIShaderPlatform, Quality);
    if (!Resource) return Finish(TEXT("ACTIVE_MATERIAL_RESOURCE_MISSING"));
    if (!Resource->IsCompilationFinished()) return Finish(TEXT("MATERIAL_COMPILATION_PENDING"));
    const FMaterialShaderMap* ShaderMap = Resource->GetGameThreadShaderMap();
    if (!ShaderMap || !ShaderMap->IsValidForRendering()) return Finish(TEXT("VALID_SHADER_MAP_MISSING"));
    const bool UsesWpo = ShaderMap->UsesWorldPositionOffset();
    const bool UsesPdo = ShaderMap->UsesPixelDepthOffset();
    const bool UsesDisplacement = ShaderMap->UsesDisplacement();
    const bool ModifiesPosition = ShaderMap->ModifiesMeshPosition();
    const FMaterialRelevance Relevance = Material->GetRelevance_Concurrent(GMaxRHIShaderPlatform);
    const float MaxWpo = Material->GetMaxWorldPositionOffsetDisplacement();
    Data->SetStringField(TEXT("data_status"), TEXT("AVAILABLE"));
    Data->SetBoolField(TEXT("uses_wpo"), UsesWpo);
    Data->SetBoolField(TEXT("uses_pdo"), UsesPdo);
    Data->SetBoolField(TEXT("uses_displacement"), UsesDisplacement);
    Data->SetBoolField(TEXT("modifies_mesh_position"), ModifiesPosition);
    Data->SetBoolField(TEXT("uses_first_person_interpolation"), bool(Relevance.bUsesFirstPersonInterpolation));
    Data->SetBoolField(TEXT("always_evaluate_wpo"), Material->ShouldAlwaysEvaluateWorldPositionOffset());
    if (FMath::IsFinite(MaxWpo)) Data->SetNumberField(TEXT("configured_max_wpo_per_axis_cm"), MaxWpo);
    if (UsesPdo || UsesDisplacement || Relevance.bUsesFirstPersonInterpolation || (ModifiesPosition && !UsesWpo))
        return Finish(TEXT("NON_WPO_DEFORMATION_NOT_BOUNDED_BY_THIS_HELPER"));
    if (!UsesWpo)
    {
        Data->SetStringField(TEXT("capability"), TEXT("NO_COMPILED_MATERIAL_DEFORMATION"));
        Data->SetNumberField(TEXT("required_material_inflation_cm"), 0);
        return Finish(TEXT("ACTIVE_SHADER_HAS_NO_GEOMETRY_DEFORMATION"));
    }
    if (GMaxRHIFeatureLevel <= ERHIFeatureLevel::ES3_1)
        return Finish(TEXT("MOBILE_WPO_CLAMP_NOT_GUARANTEED"));
    if (!FMath::IsFinite(MaxWpo) || MaxWpo <= 0)
        return Finish(TEXT("WPO_CLAMP_ZERO_OR_INVALID_IS_UNBOUNDED"));
    Data->SetStringField(TEXT("capability"), TEXT("WPO_CLAMP_CONFIGURED"));
    Data->SetBoolField(TEXT("requires_all_component_materials_and_primitive_max_wpo_extent"), true);
    return Finish(TEXT("AGGREGATE_ALL_SLOT_CLAMPS_BEFORE_WORLD_AABB_INFLATION"));
}

bool UBlindAssistCaptureLibrary::ZeroDerivedMaterialDeformation(UMaterial* Material)
{
#if WITH_EDITOR
    if (!IsInGameThread() || !Material) return false;
    const FString PackageName = Material->GetOutermost()->GetName();
    if (!PackageName.StartsWith(TEXT("/Game/CNH")) || !PackageName.Contains(TEXT("/D_"))
        || !Material->GetName().Contains(TEXT("_CNH_"))) return false;
    FExpressionInput* Wpo = Material->GetExpressionInputForProperty(MP_WorldPositionOffset);
    FExpressionInput* Pdo = Material->GetExpressionInputForProperty(MP_PixelDepthOffset);
    FExpressionInput* Attributes = Material->GetExpressionInputForProperty(MP_MaterialAttributes);
    if (!Wpo || !Pdo || !Attributes || (Material->bUseMaterialAttributes && !Attributes->Expression)) return false;
    UMaterialExpression* OriginalAttributes = Attributes->Expression;
    const int32 OriginalOutput = Attributes->OutputIndex;
    auto* ZeroVector = NewObject<UMaterialExpressionConstant3Vector>(Material);
    auto* ZeroScalar = NewObject<UMaterialExpressionConstant>(Material);
    ZeroVector->Material = Material;
    ZeroScalar->Material = Material;
    ZeroVector->Constant = FLinearColor(0, 0, 0, 0);
    ZeroScalar->R = 0.0f;
    Material->GetExpressionCollection().AddExpression(ZeroVector);
    Material->GetExpressionCollection().AddExpression(ZeroScalar);
    UMaterialExpressionSetMaterialAttributes* Wrapper = nullptr;
    if (Material->bUseMaterialAttributes)
    {
        Wrapper = NewObject<UMaterialExpressionSetMaterialAttributes>(Material);
        Wrapper->Material = Material;
        Material->GetExpressionCollection().AddExpression(Wrapper);
        // Public Engine API maintains Inputs/AttributeSetTypes atomically. Never
        // mutate the protected parallel arrays through Python property callbacks.
        if (!Wrapper->ConnectInputAttribute(MP_MaterialAttributes, OriginalAttributes, OriginalOutput)
            || !Wrapper->ConnectInputAttribute(MP_WorldPositionOffset, ZeroVector)
            || !Wrapper->ConnectInputAttribute(MP_PixelDepthOffset, ZeroScalar)) return false;
        Attributes->Connect(0, Wrapper);
        if (Wrapper->GetInput(0)->Expression != OriginalAttributes
            || Wrapper->GetInput(0)->OutputIndex != OriginalOutput
            || Wrapper->GetInput(1)->Expression != ZeroVector
            || Wrapper->GetInput(2)->Expression != ZeroScalar) return false;
    }
    Wpo->Connect(0, ZeroVector);
    Pdo->Connect(0, ZeroScalar);
    return Wpo->Expression == ZeroVector && Pdo->Expression == ZeroScalar
        && (!Wrapper || Attributes->Expression == Wrapper);
#else
    return false;
#endif
}

namespace
{
TWeakObjectPtr<USceneCaptureComponent2D> PreparedCapture;
FVector PreparedLocation;
float PreparedFov = 0.0f;
int32 PreparedWidth = 0;
}

bool UBlindAssistCaptureLibrary::PrepareCaptureReadiness(USceneCaptureComponent2D* Capture)
{
    if (!IsInGameThread()) return false;
    PreparedCapture.Reset();
    if (!Capture || !Capture->GetWorld() || !Capture->TextureTarget
        || Capture->TextureTarget->SizeX <= 0 || Capture->ProjectionType != ECameraProjectionMode::Perspective
        || Capture->bUseCustomProjectionMatrix || !FMath::IsFinite(Capture->FOVAngle)
        || Capture->FOVAngle <= 0.0f || Capture->FOVAngle >= 180.0f
        || !IStreamingManager::Get().IsStreamingEnabled()
        || !IStreamingManager::Get().IsTextureStreamingEnabled()) return false;
    // Newly compiled resources must exist before their view demand is calculated.
    if (FAssetCompilingManager::Get().GetNumRemainingAssets() != 0
        || (GShaderCompilingManager && GShaderCompilingManager->GetNumRemainingJobs() != 0)) return false;
    const FVector Location = Capture->GetComponentLocation();
    if (Location.ContainsNaN()) return false;
    const float Width = float(Capture->TextureTarget->SizeX);
    // Same perspective formula as UEditorEngine::UpdateSingleViewportClient.
    IStreamingManager::Get().AddViewInformation(Location, Width,
        Width / FMath::Tan(FMath::DegreesToRadians(Capture->FOVAngle * 0.5f)),
        1.0f, false, 0.0f, nullptr, Capture->GetWorld());
    // UE 5.8 completes its mip calculation synchronously and submits load requests here.
    // This is explicitly a full demand calculation, not the obsolete wanting-resources ID.
    IStreamingManager::Get().UpdateResourceStreaming(0.0f, true);
    PreparedCapture = Capture;
    PreparedLocation = Location;
    PreparedFov = Capture->FOVAngle;
    PreparedWidth = Capture->TextureTarget->SizeX;
    return true;
}

FBlindAssistCaptureReadiness UBlindAssistCaptureLibrary::PollCaptureReadiness(UObject* WorldContextObject)
{
    FBlindAssistCaptureReadiness Result;
    if (!IsInGameThread() || !GEngine || !WorldContextObject
        || !GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::ReturnNull))
    {
        return Result;
    }
    Result.AssetCompilationRemaining = FAssetCompilingManager::Get().GetNumRemainingAssets();
    Result.ShaderJobsRemaining = GShaderCompilingManager ? GShaderCompilingManager->GetNumRemainingJobs() : 0;
    if (!IStreamingManager::Get().IsStreamingEnabled()
        || !IStreamingManager::Get().IsTextureStreamingEnabled()) return Result;
    Result.PendingRenderAssets = 0;
    for (TObjectIterator<UStreamableRenderAsset> It; It; ++It)
    {
        if (It->HasPendingInitOrStreaming() || It->bHasStreamingUpdatePending)
            ++Result.PendingRenderAssets;
    }
    const USceneCaptureComponent2D* Capture = PreparedCapture.Get();
    Result.StreamingUpdateCompleted = Capture && Capture->TextureTarget
        && Capture->GetWorld() == GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::ReturnNull)
        && Capture->GetComponentLocation().Equals(PreparedLocation, 0.0f)
        && Capture->FOVAngle == PreparedFov && Capture->TextureTarget->SizeX == PreparedWidth;
    if (Result.AssetCompilationRemaining != 0 || Result.ShaderJobsRemaining != 0)
    {
        PreparedCapture.Reset();
        Result.StreamingUpdateCompleted = false;
    }
    if (Result.StreamingUpdateCompleted)
    {
        // Keep the actual capture view submitted while IO finishes across editor ticks.
        // A regular viewport can otherwise replace the one-frame preparation view.
        IStreamingManager::Get().AddViewInformation(PreparedLocation, float(PreparedWidth),
            float(PreparedWidth) / FMath::Tan(FMath::DegreesToRadians(PreparedFov * 0.5f)),
            1.0f, false, 0.0f, nullptr, Capture->GetWorld());
    }
    Result.ReadySupported = true;
    return Result;
}

bool UBlindAssistCaptureLibrary::ExportDepthNpy(
    UObject* WorldContextObject, UTextureRenderTarget2D* TextureRenderTarget, const FString& Filename)
{
    static_assert(PLATFORM_LITTLE_ENDIAN && sizeof(float) == 4, "NPY output requires little-endian float32");
    if (!IsInGameThread() || !TextureRenderTarget || Filename.IsEmpty()
        || TextureRenderTarget->SizeX <= 0 || TextureRenderTarget->SizeY <= 0)
    {
        UE_LOG(LogBlindAssistCapture, Error, TEXT("ExportDepthNpy: invalid target, filename, or calling thread"));
        return false;
    }

    TArray<FLinearColor> Samples;
    const int64 PixelCount = int64(TextureRenderTarget->SizeX) * TextureRenderTarget->SizeY;
    if (!UKismetRenderingLibrary::ReadRenderTargetRaw(WorldContextObject, TextureRenderTarget, Samples, false)
        || Samples.Num() != PixelCount)
    {
        UE_LOG(LogBlindAssistCapture, Error, TEXT("ExportDepthNpy: render target read failed"));
        return false;
    }

    // Match the nearfield exporters' str(dict(...)) and padding byte for byte.
    FString Header = FString::Printf(
        TEXT("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d)}"),
        TextureRenderTarget->SizeY, TextureRenderTarget->SizeX);
    const int32 Padding = (64 - (10 + Header.Len() + 1) % 64) % 64;
    Header += FString::ChrN(Padding, TEXT(' '));
    Header += TEXT("\n");
    FTCHARToUTF8 HeaderBytes(*Header);
    const uint16 HeaderLength = static_cast<uint16>(HeaderBytes.Length());
    const uint8 Prefix[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
        static_cast<uint8>(HeaderLength & 0xff), static_cast<uint8>(HeaderLength >> 8)};

    TArray<float> Depth;
    Depth.SetNumUninitialized(Samples.Num());
    for (int32 Index = 0; Index < Samples.Num(); ++Index)
    {
        const float Red = Samples[Index].R;
        // Python float division is double precision before array('f') rounds to float32.
        Depth[Index] = FMath::IsFinite(Red) && Red > 0.0f && Red < 10000.0f
            ? static_cast<float>(static_cast<double>(Red) / 100.0) : 0.0f;
    }

    IFileManager& Files = IFileManager::Get();
    const FString Output = FPaths::ConvertRelativePathToFull(Filename);
    const FString Partial = Output + TEXT(".partial");
    if (Files.FileExists(*Output))
    {
        UE_LOG(LogBlindAssistCapture, Error, TEXT("ExportDepthNpy: output already exists: %s"), *Output);
        return false;
    }
    TUniquePtr<FArchive> Writer(Files.CreateFileWriter(*Partial, FILEWRITE_NoReplaceExisting));
    if (!Writer)
    {
        UE_LOG(LogBlindAssistCapture, Error, TEXT("ExportDepthNpy: cannot open %s"), *Partial);
        return false;
    }
    Writer->Serialize(const_cast<uint8*>(Prefix), sizeof(Prefix));
    Writer->Serialize(const_cast<ANSICHAR*>(HeaderBytes.Get()), HeaderBytes.Length());
    Writer->Serialize(Depth.GetData(), int64(Depth.Num()) * sizeof(float));
    const bool bWriteSucceeded = !Writer->IsError();
    const bool bClosed = Writer->Close();
    Writer.Reset();
    if (!bWriteSucceeded || !bClosed)
    {
        Files.Delete(*Partial, false, false, true);
        UE_LOG(LogBlindAssistCapture, Error, TEXT("ExportDepthNpy: write failed for %s"), *Output);
        return false;
    }
    // Never delete an existing destination before rename: preserve it on any failure.
    if (!Files.Move(*Output, *Partial, false, false, false, true))
    {
        Files.Delete(*Partial, false, false, true);
        UE_LOG(LogBlindAssistCapture, Error, TEXT("ExportDepthNpy: rename failed for %s"), *Output);
        return false;
    }
    return true;
}
