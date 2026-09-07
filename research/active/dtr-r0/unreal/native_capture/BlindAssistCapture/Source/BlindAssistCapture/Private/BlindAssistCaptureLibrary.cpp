#include "BlindAssistCaptureLibrary.h"

#include "Engine/TextureRenderTarget2D.h"
#include "HAL/FileManager.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "Misc/Paths.h"
#include "Serialization/Archive.h"

DEFINE_LOG_CATEGORY_STATIC(LogBlindAssistCapture, Log, All);

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
