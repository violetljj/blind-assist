#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "BlindAssistCaptureLibrary.generated.h"

class UTextureRenderTarget2D;

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
};

UCLASS()
class BLINDASSISTCAPTURE_API UBlindAssistCaptureLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
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
};
