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
struct FNpyPart
{
    FString Filename;
    int32 Width = 0, Height = 0;
    bool bDepthMetres = false;
    TUniquePtr<FRHIGPUTextureReadback> Readback;
};
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
    TArray<FNpyPart> NpyParts;
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
struct FNpySlot
{
    int32 Width, Height;
    TUniquePtr<FRHIGPUTextureReadback> Readback;
};
TArray<FNpySlot> FreeNpySlots;

bool JobOwnsPath(const FJob& Job, const FString& Output)
{
    return Job->RgbFilename.Equals(Output, ESearchCase::IgnoreCase)
        || Job->DepthFilename.Equals(Output, ESearchCase::IgnoreCase)
        || Job->NpyParts.ContainsByPredicate([&Output](const FNpyPart& Part)
            { return Part.Filename.Equals(Output, ESearchCase::IgnoreCase); });
}

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

TArray64<uint8> EncodeNpy(const TArray<float>& Values, int32 Width, int32 Height, int32 Channels)
{
    FString Header = Channels == 1
        ? FString::Printf(TEXT("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d)}"), Height, Width)
        : FString::Printf(TEXT("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, %d)}"), Height, Width, Channels);
    Header += FString::ChrN((64 - (10 + Header.Len() + 1) % 64) % 64, TEXT(' '));
    Header += TEXT("\n");
    FTCHARToUTF8 HeaderBytes(*Header);
    const uint16 Length = static_cast<uint16>(HeaderBytes.Length());
    const uint8 Prefix[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, static_cast<uint8>(Length & 255), static_cast<uint8>(Length >> 8)};
    TArray64<uint8> Npy;
    Npy.Append(Prefix, sizeof(Prefix));
    Npy.Append(reinterpret_cast<const uint8*>(HeaderBytes.Get()), HeaderBytes.Length());
    Npy.Append(reinterpret_cast<const uint8*>(Values.GetData()), int64(Values.Num()) * sizeof(float));
    return Npy;
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
    TArray64<uint8> Npy = EncodeNpy(Depth, Job->Width, Job->Height, 1);
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

void CheckNpyReadback(FRHICommandListImmediate& RHICmdList, FJob Job, bool bWait)
{
    for (FNpyPart& Part : Job->NpyParts)
    {
        if (bWait) Part.Readback->Wait(RHICmdList, Part.Readback->GetLastCopyGPUMask());
        if (!Part.Readback->IsReady())
        {
            Job->State.store(EPairState::Copied);
            return;
        }
    }
    Job->GpuReadySeconds = FPlatformTime::Seconds() - Job->Started;
    const double ReadStart = FPlatformTime::Seconds();
    TArray<TArray<float>> Arrays;
    for (FNpyPart& Part : Job->NpyParts)
    {
        int32 Pitch = 0, Height = 0;
        const float* Pixels = static_cast<const float*>(Part.Readback->Lock(Pitch, &Height));
        if (!Pixels || Pitch < Part.Width || Height < Part.Height)
        {
            if (Pixels) Part.Readback->Unlock();
            // All fences are ready here: release every owned readback on render thread.
            for (FNpyPart& Release : Job->NpyParts) Release.Readback.Reset();
            Job->Error = TEXT("Invalid attribute GPU readback mapping/pitch");
            Job->State.store(EPairState::Done);
            return;
        }
        const int32 Channels = Part.bDepthMetres ? 1 : 3;
        TArray<float>& Values = Arrays.AddDefaulted_GetRef();
        Values.SetNumUninitialized(Part.Width * Part.Height * Channels);
        for (int32 Y = 0; Y < Part.Height; ++Y)
        {
            for (int32 X = 0; X < Part.Width; ++X)
            {
                const float* Source = Pixels + (int64(Y) * Pitch + X) * 4;
                float* Dest = Values.GetData() + (int64(Y) * Part.Width + X) * Channels;
                if (Part.bDepthMetres)
                    Dest[0] = FMath::IsFinite(Source[0]) && Source[0] > 0.0f && Source[0] < 10000.0f
                        ? static_cast<float>(static_cast<double>(Source[0]) / 100.0) : 0.0f;
                else
                    FMemory::Memcpy(Dest, Source, 3 * sizeof(float));
            }
        }
        Part.Readback->Unlock();
        FreeNpySlots.Add(FNpySlot{Part.Width, Part.Height, MoveTemp(Part.Readback)});
    }
    Job->ReadbackSeconds = FPlatformTime::Seconds() - ReadStart;
    Job->Writer = Async(EAsyncExecution::ThreadPool, [Job, Arrays = MoveTemp(Arrays)]() mutable
    {
        for (int32 Index = 0; Index < Arrays.Num(); ++Index)
        {
            const FNpyPart& Part = Job->NpyParts[Index];
            const double EncodeStart = FPlatformTime::Seconds();
            TArray64<uint8> Npy = EncodeNpy(Arrays[Index], Part.Width, Part.Height, Part.bDepthMetres ? 1 : 3);
            Job->EncodeSeconds += FPlatformTime::Seconds() - EncodeStart;
            const double WriteStart = FPlatformTime::Seconds();
            const FString Partial = Part.Filename + TEXT(".partial");
            if (!WritePartial(Partial, Npy.GetData(), Npy.Num()))
            {
                Job->Error = TEXT("Attribute partial write failed: ") + Part.Filename;
                break;
            }
            if (!IFileManager::Get().Move(*Part.Filename, *Partial, false, false, false, true))
            {
                IFileManager::Get().Delete(*Partial, false, false, true);
                Job->Error = TEXT("Attribute rename failed: ") + Part.Filename;
                break;
            }
            Job->WriteSeconds += FPlatformTime::Seconds() - WriteStart;
        }
    });
    Job->State.store(EPairState::Writing);
}

void CheckReadback(FRHICommandListImmediate& RHICmdList, FJob Job, bool bWait)
{
    if (!Job->NpyParts.IsEmpty())
    {
        CheckNpyReadback(RHICmdList, Job, bWait);
        return;
    }
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
            || Jobs.ContainsByPredicate([&Output](const FJob& Job) { return JobOwnsPath(Job, Output); }))
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

bool UBlindAssistCaptureLibrary::SubmitCaptureNpyBatch(const TArray<UTextureRenderTarget2D*>& Targets,
    const TArray<FString>& Filenames, const TArray<bool>& DepthMetres, int32 MaxPending)
{
    static_assert(PLATFORM_LITTLE_ENDIAN && sizeof(float) == 4, "NPY output requires little-endian float32");
    check(IsInGameThread());
    const double SubmitStart = FPlatformTime::Seconds();
    PumpPairs(false);
    if (MaxPending <= 0 || Targets.Num() < 1 || Targets.Num() > 4
        || Filenames.Num() != Targets.Num() || DepthMetres.Num() != Targets.Num())
        return PairReject(TEXT("Invalid attribute batch dimensions"));
    if (Jobs.Num() >= MaxPending) return false;
    FJob Job = MakeShared<FPairJob, ESPMode::ThreadSafe>();
    TArray<FTextureRHIRef> Textures;
    for (int32 Index = 0; Index < Targets.Num(); ++Index)
    {
        UTextureRenderTarget2D* Target = Targets[Index];
        if (!Target || !Target->GetResource() || Target->SizeX <= 0 || Target->SizeY <= 0 || Filenames[Index].IsEmpty())
            return PairReject(TEXT("Invalid attribute target/filename"));
        FTextureRHIRef Texture = Target->GetResource()->TextureRHI;
        if (!Texture || Texture->GetFormat() != PF_A32B32G32R32F || Texture->IsMultisampled()
            || int64(Target->SizeX) * Target->SizeY * 3 > MAX_int32)
            return PairReject(TEXT("Attribute batch requires bounded RGBA32F targets"));
        const FString Output = FPaths::ConvertRelativePathToFull(Filenames[Index]);
        if (!IFileManager::Get().DirectoryExists(*FPaths::GetPath(Output))
            || IFileManager::Get().FileExists(*Output) || IFileManager::Get().FileExists(*(Output + TEXT(".partial")))
            || JobOwnsPath(Job, Output)
            || Jobs.ContainsByPredicate([&Output](const FJob& Pending) { return JobOwnsPath(Pending, Output); }))
            return PairReject(TEXT("Attribute output exists, is pending, or parent is missing: ") + Output);
        FNpyPart& Part = Job->NpyParts.AddDefaulted_GetRef();
        Part.Filename = Output; Part.Width = Target->SizeX; Part.Height = Target->SizeY;
        Part.bDepthMetres = DepthMetres[Index];
        Textures.Add(Texture);
    }
    Job->Started = FPlatformTime::Seconds();
    Jobs.Add(Job);
    ENQUEUE_RENDER_COMMAND(BlindAssistSubmitNpyBatch)([Job, Textures = MoveTemp(Textures)](FRHICommandListImmediate& RHICmdList)
    {
        for (int32 Index = 0; Index < Job->NpyParts.Num(); ++Index)
        {
            FNpyPart& Part = Job->NpyParts[Index];
            const int32 SlotIndex = FreeNpySlots.IndexOfByPredicate([&Part](const FNpySlot& Slot)
                { return Slot.Width == Part.Width && Slot.Height == Part.Height; });
            if (SlotIndex != INDEX_NONE)
            {
                Part.Readback = MoveTemp(FreeNpySlots[SlotIndex].Readback);
                FreeNpySlots.RemoveAtSwap(SlotIndex);
            }
            else
            {
                // Bound stale staging memory if callers change dimensions.
                FreeNpySlots.Empty();
                Part.Readback = MakeUnique<FRHIGPUTextureReadback>(TEXT("BlindAssistAttribute"));
            }
            RHICmdList.Transition(FRHITransitionInfo(Textures[Index], ERHIAccess::Unknown, ERHIAccess::CopySrc));
            Part.Readback->EnqueueCopy(RHICmdList, Textures[Index]);
            RHICmdList.Transition(FRHITransitionInfo(Textures[Index], ERHIAccess::CopySrc, ERHIAccess::SRVMask));
        }
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

bool UBlindAssistCaptureLibrary::ResetCapturePairs()
{
    if (!IsInGameThread() || !Jobs.IsEmpty()) return false;
    PairProfile = FBlindAssistRgbWriteProfile();
    return true;
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
    ENQUEUE_RENDER_COMMAND(BlindAssistReleasePairSlots)([](FRHICommandListImmediate&) { FreeSlots.Empty(); FreeNpySlots.Empty(); });
    FlushRenderingCommands();
    return PairProfile;
}
