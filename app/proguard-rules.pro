# Keep TensorFlow Lite JNI entry points.
-keep class org.tensorflow.lite.** { *; }
-keep class com.linnan.blindassist.vision.Dav2SnpeDepthEstimator { public <init>(android.app.Application, java.io.File); }
-keepclasseswithmembernames class com.linnan.blindassist.vision.Dav2SnpeDepthEstimator { native <methods>; }
