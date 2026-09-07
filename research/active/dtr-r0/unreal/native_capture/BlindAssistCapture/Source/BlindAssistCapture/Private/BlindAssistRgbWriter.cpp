#include "BlindAssistCaptureLibrary.h"

#include "Async/Async.h"
#include "Engine/TextureRenderTarget2D.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformTime.h"
#include "ImageCore.h"
#include "ImageUtils.h"
#include "IImageWrapperModule.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "Serialization/Archive.h"

namespace
{
struct FWriteResult
{
    FString Error;
    double EncodeSeconds = 0.0;
    double WriteSeconds = 0.0;
};

struct FPendingWrite
{
    FString Filename;
    TFuture<FWriteResult> Future;
};

// Submission and reaping are game-thread-only; workers own images and return values.
TArray<FPendingWrite> PendingWrites;
FBlindAssistRgbWriteProfile Profile;

FBlindAssistRgbWriteProfile Reap(bool bWait)
{
    check(IsInGameThread());
    for (int32 Index = PendingWrites.Num() - 1; Index >= 0; --Index)
    {
        auto& Entry = PendingWrites[Index];
        if (bWait || Entry.Future.IsReady())
        {
            const FWriteResult Result = Entry.Future.Get();
            Profile.EncodeSeconds += Result.EncodeSeconds;
            Profile.WriteSeconds += Result.WriteSeconds;
            if (Result.Error.IsEmpty())
            {
                ++Profile.Completed;
            }
            else
            {
                ++Profile.Failed;
                Profile.LastError = Result.Error;
            }
            PendingWrites.RemoveAtSwap(Index);
        }
    }
    Profile.Pending = PendingWrites.Num();
    return Profile;
}

bool Reject(const FString& Error)
{
    ++Profile.Failed;
    Profile.LastError = Error;
    return false;
}

FWriteResult EncodeAndWrite(FImage Image, FString Filename)
{
    FWriteResult Result;
    TArray64<uint8> Compressed;
    const double EncodeStart = FPlatformTime::Seconds();
    // Same lossless compressor/default quality as ExportRenderTarget2DAsPNG.
    const bool bEncoded = FImageUtils::CompressImage(Compressed, TEXT("PNG"), Image);
    Result.EncodeSeconds = FPlatformTime::Seconds() - EncodeStart;
    if (!bEncoded)
    {
        Result.Error = TEXT("PNG encode failed: ") + Filename;
        return Result;
    }

    const double WriteStart = FPlatformTime::Seconds();
    IFileManager& Files = IFileManager::Get();
    const FString Partial = Filename + TEXT(".partial");
    TUniquePtr<FArchive> Writer(Files.CreateFileWriter(*Partial, FILEWRITE_NoReplaceExisting));
    if (!Writer)
    {
        Result.Error = TEXT("Cannot create PNG partial: ") + Partial;
    }
    else
    {
        Writer->Serialize(Compressed.GetData(), Compressed.Num());
        const bool bWritten = !Writer->IsError();
        const bool bClosed = Writer->Close();
        Writer.Reset();
        if (!bWritten || !bClosed || !Files.Move(*Filename, *Partial, false, false, false, true))
        {
            Files.Delete(*Partial, false, false, true);
            Result.Error = TEXT("PNG write or rename failed: ") + Filename;
        }
    }
    Result.WriteSeconds = FPlatformTime::Seconds() - WriteStart;
    return Result;
}
}

bool UBlindAssistCaptureLibrary::ExportRgbPng(
    UObject* WorldContextObject, UTextureRenderTarget2D* TextureRenderTarget,
    const FString& Filename, int32 MaxPending)
{
    check(IsInGameThread());
    Reap(false);
    if (!TextureRenderTarget || !TextureRenderTarget->GetResource()
        || Filename.IsEmpty() || MaxPending <= 0)
    {
        return Reject(TEXT("Invalid RGB target, filename, or queue bound"));
    }
    if (PendingWrites.Num() >= MaxPending)
    {
        return false; // Backpressure is not an export failure.
    }
    if (TextureRenderTarget->GetFormat() != PF_B8G8R8A8 && TextureRenderTarget->GetFormat() != PF_R8G8B8A8)
    {
        return Reject(TEXT("RGB export requires an 8-bit RGBA render target"));
    }
    const FString Output = FPaths::ConvertRelativePathToFull(Filename);
    if (IFileManager::Get().FileExists(*Output) || IFileManager::Get().FileExists(*(Output + TEXT(".partial")))
        || PendingWrites.ContainsByPredicate([&Output](const FPendingWrite& Entry) { return Entry.Filename == Output; }))
    {
        return Reject(TEXT("RGB output already exists or is pending: ") + Output);
    }
    // Load on the game thread before any worker enters FImageUtils compression.
    FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
    FImage Image;
    const double ReadStart = FPlatformTime::Seconds();
    const bool bRead = FImageUtils::GetRenderTargetImage(TextureRenderTarget, Image);
    Profile.ReadbackSeconds += FPlatformTime::Seconds() - ReadStart;
    if (!bRead)
    {
        return Reject(TEXT("RGB readback failed: ") + Output);
    }
    FPendingWrite Entry;
    Entry.Filename = Output;
    Entry.Future = Async(EAsyncExecution::ThreadPool,
        [Image = MoveTemp(Image), Output]() mutable { return EncodeAndWrite(MoveTemp(Image), Output); });
    PendingWrites.Add(MoveTemp(Entry));
    ++Profile.Submitted;
    Profile.Pending = PendingWrites.Num();
    Profile.PeakPending = FMath::Max(Profile.PeakPending, Profile.Pending);
    return true;
}

FBlindAssistRgbWriteProfile UBlindAssistCaptureLibrary::PollRgbWrites()
{
    return Reap(false);
}

FBlindAssistRgbWriteProfile UBlindAssistCaptureLibrary::DrainRgbWrites()
{
    return Reap(true);
}

bool UBlindAssistCaptureLibrary::ResetRgbWrites()
{
    if (!IsInGameThread() || !PendingWrites.IsEmpty()) return false;
    Profile = FBlindAssistRgbWriteProfile();
    return true;
}
