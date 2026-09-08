using UnrealBuildTool;

public class BlindAssistCapture : ModuleRules
{
    public BlindAssistCapture(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new[] { "Core", "CoreUObject", "Engine" });
        PrivateDependencyModuleNames.AddRange(new[] { "ImageCore", "ImageWrapper", "RHI", "RenderCore", "Json" });
    }
}
