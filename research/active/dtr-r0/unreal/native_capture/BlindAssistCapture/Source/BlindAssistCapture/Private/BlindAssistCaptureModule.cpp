#include "Modules/ModuleManager.h"
#include "BlindAssistCaptureLibrary.h"

class FBlindAssistCaptureModule : public IModuleInterface
{
public:
    virtual void ShutdownModule() override
    {
        UBlindAssistCaptureLibrary::DrainRgbWrites();
    }
};

IMPLEMENT_MODULE(FBlindAssistCaptureModule, BlindAssistCapture)
