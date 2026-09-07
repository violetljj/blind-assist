#include "Modules/ModuleManager.h"
#include "BlindAssistCaptureLibrary.h"

class FBlindAssistCaptureModule : public IModuleInterface
{
public:
    virtual void ShutdownModule() override
    {
        UBlindAssistCaptureLibrary::DrainRgbWrites();
        UBlindAssistCaptureLibrary::DrainCapturePairs();
    }
};

IMPLEMENT_MODULE(FBlindAssistCaptureModule, BlindAssistCapture)
