#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "BlindAssistCaptureLibrary.generated.h"

class UTextureRenderTarget2D;

UCLASS()
class BLINDASSISTCAPTURE_API UBlindAssistCaptureLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    /** Export red-channel centimetres as the capture pipeline's row-major float32 metres. */
    UFUNCTION(BlueprintCallable, Category = "BlindAssist|Capture", meta = (WorldContext = "WorldContextObject"))
    static bool ExportDepthNpy(UObject* WorldContextObject, UTextureRenderTarget2D* TextureRenderTarget, const FString& Filename);
};
