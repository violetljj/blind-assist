#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "BlindAssistCaptureLibrary.generated.h"

class UTextureRenderTarget2D;
class USceneCaptureComponent2D;

/** Nonblocking editor readiness observation; not a visual-convergence guarantee. */
USTRUCT(BlueprintType)
struct BLINDASSISTCAPTURE_API FBlindAssistCaptureReadiness
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    bool ReadySupported = false;
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    int32 AssetCompilationRemaining = -1;
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    int32 ShaderJobsRemaining = -1;
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    int32 PendingRenderAssets = -1;
    /** A synchronous demand calculation completed for the prepared capture view. */
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    bool StreamingUpdateCompleted = false;
};

USTRUCT(BlueprintType)
struct BLINDASSISTCAPTURE_API FBlindAssistRgbWriteProfile
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    int32 Pending = 0;
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    int32 PeakPending = 0;
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    int32 Submitted = 0;
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    int32 Completed = 0;
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    int32 Failed = 0;
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    FString LastError;
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    double ReadbackSeconds = 0.0;
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    double EncodeSeconds = 0.0;
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    double WriteSeconds = 0.0;
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    double SubmitSeconds = 0.0;
    /** Submission to observed GPU readiness, including polling delay. */
    UPROPERTY(BlueprintReadOnly, Category = "BlindAssist|Capture")
    double GpuReadySeconds = 0.0;
};

UCLASS()
class BLINDASSISTCAPTURE_API UBlindAssistCaptureLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    /** Read-only loaded HLOD source-actor mappings. Writes only a fresh path under BA_CITY_OUT.
     * True means JSON export succeeded, never source/visibility admission. Does not load source actors. */
    UFUNCTION(BlueprintCallable, Category = "BlindAssist|Capture", meta = (WorldContext = "WorldContextObject"))
    static bool ExportWorldHLODSourceMembership(UObject* WorldContextObject, const FString& Filename);

    /** Submit the actual perspective capture view and synchronously calculate streaming demand.
     * Returns false while compilation remains or the capture is unsupported; retry across ticks.
     * Does not wait for requested mip IO, and does not cover temporal/virtual-texture convergence. */
    UFUNCTION(BlueprintCallable, Category = "BlindAssist|Capture")
    static bool PrepareCaptureReadiness(USceneCaptureComponent2D* Capture);

    /** Poll actual initialization/streaming requests after PrepareCaptureReadiness succeeds.
     * Counts are process-wide. A changed camera requires a new preparation. */
    UFUNCTION(BlueprintCallable, Category = "BlindAssist|Capture", meta = (WorldContext = "WorldContextObject"))
    static FBlindAssistCaptureReadiness PollCaptureReadiness(UObject* WorldContextObject);

    /** Export red-channel centimetres as the capture pipeline's row-major float32 metres. */
    UFUNCTION(BlueprintCallable, Category = "BlindAssist|Capture", meta = (WorldContext = "WorldContextObject"))
    static bool ExportDepthNpy(UObject* WorldContextObject, UTextureRenderTarget2D* TextureRenderTarget, const FString& Filename);

    /** Read the same pixels as RenderingLibrary export, then encode/write PNG off-thread.
     * False means rejected or failed; poll before submitting when the queue is full.
     * Filename and its parent directory must be unique/existing respectively. */
    UFUNCTION(BlueprintCallable, Category = "BlindAssist|Capture", meta = (WorldContext = "WorldContextObject"))
    static bool ExportRgbPng(UObject* WorldContextObject, UTextureRenderTarget2D* TextureRenderTarget, const FString& Filename, int32 MaxPending = 4);

    /** Reap ready writes; cumulative timings cover this process, in seconds. */
    UFUNCTION(BlueprintCallable, Category = "BlindAssist|Capture")
    static FBlindAssistRgbWriteProfile PollRgbWrites();

    /** Wait for all queued writes; callers must check Failed as well as Pending. */
    UFUNCTION(BlueprintCallable, Category = "BlindAssist|Capture")
    static FBlindAssistRgbWriteProfile DrainRgbWrites();

    /** Reset cumulative counters only after the caller has polled/drained all writes. */
    UFUNCTION(BlueprintCallable, Category = "BlindAssist|Capture")
    static bool ResetRgbWrites();

    /** Queue GPU copies after the caller's CaptureScene commands. No synchronous readback. */
    UFUNCTION(BlueprintCallable, Category = "BlindAssist|Capture")
    static bool SubmitCapturePair(UTextureRenderTarget2D* RgbTarget, UTextureRenderTarget2D* DepthTarget, const FString& RgbFilename, const FString& DepthFilename, int32 MaxPending = 4);

    /** Schedule GPU fence checks and reap completed pairs. */
    UFUNCTION(BlueprintCallable, Category = "BlindAssist|Capture")
    static FBlindAssistRgbWriteProfile PollCapturePairs();

    /** Final drain may block on GPU fences and CPU writers. Check Failed before accepting output. */
    UFUNCTION(BlueprintCallable, Category = "BlindAssist|Capture")
    static FBlindAssistRgbWriteProfile DrainCapturePairs();

    /** Reset cumulative counters only after the caller has polled/drained all pairs. */
    UFUNCTION(BlueprintCallable, Category = "BlindAssist|Capture")
    static bool ResetCapturePairs();
};
