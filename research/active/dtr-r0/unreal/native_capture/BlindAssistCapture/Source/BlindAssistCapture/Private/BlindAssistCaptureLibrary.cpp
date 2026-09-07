#include "BlindAssistCaptureLibrary.h"

#include "AssetCompilingManager.h"
#include "ContentStreaming.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/Engine.h"
#include "Engine/StreamableRenderAsset.h"
#include "Engine/TextureRenderTarget2D.h"
#include "HAL/FileManager.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "Misc/Paths.h"
#include "Serialization/Archive.h"
#include "ShaderCompiler.h"
#include "UObject/UObjectIterator.h"

DEFINE_LOG_CATEGORY_STATIC(LogBlindAssistCapture, Log, All);

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
