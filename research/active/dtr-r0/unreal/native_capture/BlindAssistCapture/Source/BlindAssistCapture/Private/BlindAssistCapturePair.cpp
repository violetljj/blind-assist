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
#include "RenderingThread.h"
#include "RHIGPUReadback.h"
#include "Serialization/Archive.h"
#include "TextureResource.h"
#include <atomic>

namespace
{
enum class EPairState { Queued, Copied, Checking, Writing, Done };
struct FPairJob
{
    FString RgbFilename, DepthFilename, Error;
    int32 Width = 0, Height = 0;
    EPixelFormat RgbFormat = PF_Unknown;
    EGammaSpace Gamma = EGammaSpace::sRGB;
    std::atomic<EPairState> State{EPairState::Queued};
    TUniquePtr<FRHIGPUTextureReadback> RgbReadback, DepthReadback;
    TFuture<void> Writer;
    double Started = 0.0, GpuReadySeconds = 0.0, ReadbackSeconds = 0.0;
    double EncodeSeconds = 0.0, WriteSeconds = 0.0;
};
using FJob = TSharedPtr<FPairJob, ESPMode::ThreadSafe>;
TArray<FJob> Jobs;
FBlindAssistRgbWriteProfile PairProfile;
struct FReadbackSlot
{
    int32 Width, Height;
    EPixelFormat RgbFormat;
    TUniquePtr<FRHIGPUTextureReadback> Rgb, Depth;
};
// Render-thread-only ring slots, recycled once GPU and CPU copies have finished.
TArray<FReadbackSlot> FreeSlots;

bool PairReject(const FString& Error)
{
    ++PairProfile.Failed;
    PairProfile.LastError = Error;
    return false;
}

bool WritePartial(const FString& Filename, const void* Data, int64 Bytes)
{
    TUniquePtr<FArchive> Writer(IFileManager::Get().CreateFileWriter(*Filename, FILEWRITE_NoReplaceExisting));
    if (!Writer)
    {
        return false;
    }
    Writer->Serialize(const_cast<void*>(Data), Bytes);
    const bool bWritten = !Writer->IsError();
    const bool bClosed = Writer->Close();
    Writer.Reset();
    if (!bWritten || !bClosed)
    {
        IFileManager::Get().Delete(*Filename, false, false, true);
        return false;
    }
    return true;
}

void WritePair(FJob Job, FImage Rgb, TArray<float> Depth)
{
    const double EncodeStart = FPlatformTime::Seconds();
    TArray64<uint8> Png;
    if (!FImageUtils::CompressImage(Png, TEXT("PNG"), Rgb))
    {
        Job->Error = TEXT("Pair PNG encode failed: ") + Job->RgbFilename;
        return;
    }
    FString Header = FString::Printf(TEXT("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d)}"), Job->Height, Job->Width);
    Header += FString::ChrN((64 - (10 + Header.Len() + 1) % 64) % 64, TEXT(' '));
    Header += TEXT("\n");
    FTCHARToUTF8 HeaderBytes(*Header);
    const uint16 Length = static_cast<uint16>(HeaderBytes.Length());
    const uint8 Prefix[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, static_cast<uint8>(Length & 255), static_cast<uint8>(Length >> 8)};
    TArray64<uint8> Npy;
    Npy.Append(Prefix, sizeof(Prefix));
    Npy.Append(reinterpret_cast<const uint8*>(HeaderBytes.Get()), HeaderBytes.Length());
    Npy.Append(reinterpret_cast<const uint8*>(Depth.GetData()), int64(Depth.Num()) * sizeof(float));
    Job->EncodeSeconds = FPlatformTime::Seconds() - EncodeStart;

    const double WriteStart = FPlatformTime::Seconds();
    IFileManager& Files = IFileManager::Get();
    const FString RgbPartial = Job->RgbFilename + TEXT(".partial");
    const FString DepthPartial = Job->DepthFilename + TEXT(".partial");
    const bool bRgbWritten = WritePartial(RgbPartial, Png.GetData(), Png.Num());
    const bool bDepthWritten = bRgbWritten && WritePartial(DepthPartial, Npy.GetData(), Npy.Num());
    if (!bRgbWritten || !bDepthWritten)
    {
        if (bRgbWritten) Files.Delete(*RgbPartial, false, false, true);
        Job->Error = TEXT("Pair partial write failed: ") + Job->RgbFilename;
    }
    else
    {
        const bool bRgbCommitted = Files.Move(*Job->RgbFilename, *RgbPartial, false, false, false, true);
        const bool bDepthCommitted = bRgbCommitted && Files.Move(*Job->DepthFilename, *DepthPartial, false, false, false, true);
        if (!bRgbCommitted || !bDepthCommitted)
        {
            Files.Delete(*RgbPartial, false, false, true);
            Files.Delete(*DepthPartial, false, false, true);
            Job->Error = TEXT("Pair rename failed: ") + Job->RgbFilename;
        }
    }
    Job->WriteSeconds = FPlatformTime::Seconds() - WriteStart;
}

void CheckReadback(FRHICommandListImmediate& RHICmdList, FJob Job, bool bWait)
{
    if (bWait)
    {
        Job->RgbReadback->Wait(RHICmdList, Job->RgbReadback->GetLastCopyGPUMask());
        Job->DepthReadback->Wait(RHICmdList, Job->DepthReadback->GetLastCopyGPUMask());
    }
    if (!Job->RgbReadback->IsReady() || !Job->DepthReadback->IsReady())
    {
        Job->State.store(EPairState::Copied);
        return;
    }
    Job->GpuReadySeconds = FPlatformTime::Seconds() - Job->Started;
    const double ReadStart = FPlatformTime::Seconds();
    int32 RgbPitch = 0, DepthPitch = 0, RgbHeight = 0, DepthHeight = 0;
    const uint8* RgbPixels = static_cast<const uint8*>(Job->RgbReadback->Lock(RgbPitch, &RgbHeight));
    const float* DepthPixels = static_cast<const float*>(Job->DepthReadback->Lock(DepthPitch, &DepthHeight));
    if (!RgbPixels || !DepthPixels || RgbPitch < Job->Width || DepthPitch < Job->Width
        || RgbHeight < Job->Height || DepthHeight < Job->Height)
    {
        if (RgbPixels) Job->RgbReadback->Unlock();
        if (DepthPixels) Job->DepthReadback->Unlock();
        Job->Error = TEXT("Invalid GPU readback mapping/pitch");
        Job->State.store(EPairState::Done);
        return;
    }
    FImage Rgb;
    Rgb.Init(Job->Width, Job->Height, ERawImageFormat::BGRA8, Job->Gamma);
    TArray<float> Depth;
    Depth.SetNumUninitialized(Job->Width * Job->Height);
    for (int32 Y = 0; Y < Job->Height; ++Y)
    {
        uint8* Dest = Rgb.RawData.GetData() + int64(Y) * Job->Width * 4;
        const uint8* Src = RgbPixels + int64(Y) * RgbPitch * 4;
        FMemory::Memcpy(Dest, Src, int64(Job->Width) * 4);
        for (int32 X = 0; X < Job->Width; ++X)
        {
            if (Job->RgbFormat == PF_R8G8B8A8)
            {
                Swap(Dest[X * 4], Dest[X * 4 + 2]);
            }
            const float Red = DepthPixels[(int64(Y) * DepthPitch + X) * 4];
            Depth[Y * Job->Width + X] = FMath::IsFinite(Red) && Red > 0.0f && Red < 10000.0f
                ? static_cast<float>(static_cast<double>(Red) / 100.0) : 0.0f;
        }
    }
    Job->RgbReadback->Unlock();
    Job->DepthReadback->Unlock();
    FReadbackSlot Slot{Job->Width, Job->Height, Job->RgbFormat,
        MoveTemp(Job->RgbReadback), MoveTemp(Job->DepthReadback)};
    FreeSlots.Add(MoveTemp(Slot));
    Job->ReadbackSeconds = FPlatformTime::Seconds() - ReadStart;
    Job->Writer = Async(EAsyncExecution::ThreadPool,
        [Job, Rgb = MoveTemp(Rgb), Depth = MoveTemp(Depth)]() mutable { WritePair(Job, MoveTemp(Rgb), MoveTemp(Depth)); });
    Job->State.store(EPairState::Writing);
}

FBlindAssistRgbWriteProfile PumpPairs(bool bWait)
{
    check(IsInGameThread());
    for (int32 Index = Jobs.Num() - 1; Index >= 0; --Index)
    {
        FJob Job = Jobs[Index];
        if (Job->State.load() == EPairState::Writing && Job->Writer.IsReady())
        {
            Job->Writer.Get();
            Job->State.store(EPairState::Done);
        }
        if (Job->State.load() == EPairState::Done)
        {
            PairProfile.GpuReadySeconds += Job->GpuReadySeconds;
            PairProfile.ReadbackSeconds += Job->ReadbackSeconds;
            PairProfile.EncodeSeconds += Job->EncodeSeconds;
            PairProfile.WriteSeconds += Job->WriteSeconds;
            if (Job->Error.IsEmpty()) ++PairProfile.Completed;
            else { ++PairProfile.Failed; PairProfile.LastError = Job->Error; }
            Jobs.RemoveAtSwap(Index);
        }
        else
        {
            EPairState Expected = EPairState::Copied;
            if (Job->State.compare_exchange_strong(Expected, EPairState::Checking))
            {
                ENQUEUE_RENDER_COMMAND(BlindAssistCheckPair)([Job, bWait](FRHICommandListImmediate& RHICmdList) { CheckReadback(RHICmdList, Job, bWait); });
            }
        }
    }
    PairProfile.Pending = Jobs.Num();
    return PairProfile;
}
}

bool UBlindAssistCaptureLibrary::SubmitCapturePair(UTextureRenderTarget2D* RgbTarget,
    UTextureRenderTarget2D* DepthTarget, const FString& RgbFilename, const FString& DepthFilename, int32 MaxPending)
{
    static_assert(PLATFORM_LITTLE_ENDIAN && sizeof(float) == 4, "Pair output requires little-endian float32");
    check(IsInGameThread());
    const double SubmitStart = FPlatformTime::Seconds();
    PumpPairs(false);
    if (MaxPending <= 0 || !RgbTarget || !DepthTarget || !RgbTarget->GetResource() || !DepthTarget->GetResource()
        || RgbFilename.IsEmpty() || DepthFilename.IsEmpty()) return PairReject(TEXT("Invalid capture pair input"));
    if (Jobs.Num() >= MaxPending) return false;
    FTextureRHIRef RgbTexture = RgbTarget->GetResource()->TextureRHI;
    FTextureRHIRef DepthTexture = DepthTarget->GetResource()->TextureRHI;
    if (!RgbTexture || !DepthTexture || RgbTarget->SizeX <= 0 || RgbTarget->SizeY <= 0
        || RgbTarget->SizeX != DepthTarget->SizeX || RgbTarget->SizeY != DepthTarget->SizeY
        || (RgbTexture->GetFormat() != PF_B8G8R8A8 && RgbTexture->GetFormat() != PF_R8G8B8A8)
        || DepthTexture->GetFormat() != PF_A32B32G32R32F || RgbTexture->IsMultisampled() || DepthTexture->IsMultisampled())
        return PairReject(TEXT("Capture pair requires matching RGBA8 RGB / RGBA32F depth targets"));
    const FString RgbOutput = FPaths::ConvertRelativePathToFull(RgbFilename);
    const FString DepthOutput = FPaths::ConvertRelativePathToFull(DepthFilename);
    if (RgbOutput.Equals(DepthOutput, ESearchCase::IgnoreCase)) return PairReject(TEXT("Pair filenames must differ"));
    for (const FString& Output : {RgbOutput, DepthOutput})
    {
        if (IFileManager::Get().FileExists(*Output) || IFileManager::Get().FileExists(*(Output + TEXT(".partial")))
            || Jobs.ContainsByPredicate([&Output](const FJob& Job) { return Job->RgbFilename.Equals(Output, ESearchCase::IgnoreCase) || Job->DepthFilename.Equals(Output, ESearchCase::IgnoreCase); }))
            return PairReject(TEXT("Pair output exists or is pending: ") + Output);
    }
    FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
    FJob Job = MakeShared<FPairJob, ESPMode::ThreadSafe>();
    Job->RgbFilename = RgbOutput; Job->DepthFilename = DepthOutput;
    Job->Width = RgbTarget->SizeX; Job->Height = RgbTarget->SizeY;
    Job->RgbFormat = RgbTexture->GetFormat();
    Job->Gamma = RgbTarget->IsSRGB() ? EGammaSpace::sRGB : EGammaSpace::Linear;
    Job->Started = FPlatformTime::Seconds();
    Jobs.Add(Job);
    ENQUEUE_RENDER_COMMAND(BlindAssistSubmitPair)([Job, RgbTexture, DepthTexture](FRHICommandListImmediate& RHICmdList)
    {
        const int32 SlotIndex = FreeSlots.IndexOfByPredicate([Job](const FReadbackSlot& Slot)
        { return Slot.Width == Job->Width && Slot.Height == Job->Height && Slot.RgbFormat == Job->RgbFormat; });
        if (SlotIndex != INDEX_NONE)
        {
            Job->RgbReadback = MoveTemp(FreeSlots[SlotIndex].Rgb);
            Job->DepthReadback = MoveTemp(FreeSlots[SlotIndex].Depth);
            FreeSlots.RemoveAtSwap(SlotIndex);
        }
        else
        {
            // A dimension/format change must not accumulate unused staging allocations.
            FreeSlots.Empty();
            Job->RgbReadback = MakeUnique<FRHIGPUTextureReadback>(TEXT("BlindAssistRgb"));
            Job->DepthReadback = MakeUnique<FRHIGPUTextureReadback>(TEXT("BlindAssistDepth"));
        }
        RHICmdList.Transition(FRHITransitionInfo(RgbTexture, ERHIAccess::Unknown, ERHIAccess::CopySrc));
        RHICmdList.Transition(FRHITransitionInfo(DepthTexture, ERHIAccess::Unknown, ERHIAccess::CopySrc));
        Job->RgbReadback->EnqueueCopy(RHICmdList, RgbTexture);
        Job->DepthReadback->EnqueueCopy(RHICmdList, DepthTexture);
        RHICmdList.Transition(FRHITransitionInfo(RgbTexture, ERHIAccess::CopySrc, ERHIAccess::SRVMask));
        RHICmdList.Transition(FRHITransitionInfo(DepthTexture, ERHIAccess::CopySrc, ERHIAccess::SRVMask));
        Job->State.store(EPairState::Copied);
    });
    ++PairProfile.Submitted;
    PairProfile.Pending = Jobs.Num();
    PairProfile.PeakPending = FMath::Max(PairProfile.PeakPending, PairProfile.Pending);
    PairProfile.SubmitSeconds += FPlatformTime::Seconds() - SubmitStart;
    return true;
}

FBlindAssistRgbWriteProfile UBlindAssistCaptureLibrary::PollCapturePairs()
{
    return PumpPairs(false);
}

FBlindAssistRgbWriteProfile UBlindAssistCaptureLibrary::DrainCapturePairs()
{
    check(IsInGameThread());
    while (!Jobs.IsEmpty())
    {
        // Only final drain flushes; normal submission/poll never waits for GPU work.
        FlushRenderingCommands();
        PumpPairs(true);
        FlushRenderingCommands();
        for (const FJob& Job : Jobs)
        {
            if (Job->Writer.IsValid()) Job->Writer.Wait();
        }
        PumpPairs(false);
    }
    ENQUEUE_RENDER_COMMAND(BlindAssistReleasePairSlots)([](FRHICommandListImmediate&) { FreeSlots.Empty(); });
    FlushRenderingCommands();
    return PairProfile;
}
