package com.linnan.blindassist.ui.compose

import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.activity.compose.BackHandler
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ArrowBack
import androidx.compose.material.icons.rounded.Bluetooth
import androidx.compose.material.icons.rounded.BugReport
import androidx.compose.material.icons.rounded.CameraAlt
import androidx.compose.material.icons.rounded.ChevronRight
import androidx.compose.material.icons.rounded.Favorite
import androidx.compose.material.icons.rounded.Home
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.Person
import androidx.compose.material.icons.rounded.PhoneAndroid
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.Shield
import androidx.compose.material.icons.rounded.Tune
import androidx.compose.material.icons.rounded.Vibration
import androidx.compose.material.icons.rounded.Visibility
import androidx.compose.material.icons.rounded.VolumeUp
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Shapes
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.Typography
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.localization.LocalizedText
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.ui.DetectionOverlayView
import kotlinx.coroutines.delay


@Composable
fun BlindAssistTheme(content: @Composable () -> Unit) {
    val colors = androidx.compose.material3.lightColorScheme(
        primary = BaHomeGreen,
        onPrimary = BaHomeOnAction,
        secondary = BaHomeCobalt,
        onSecondary = BaHomeOnAction,
        tertiary = BaHomeAmber,
        background = BaHomeBackground,
        surface = BaHomeSurface,
        surfaceVariant = BaHomeControlRail,
        outline = BaHomeHairline,
        outlineVariant = BaHomeHairline.copy(alpha = 0.72f),
        primaryContainer = BaHomeNavIndicator,
        onPrimaryContainer = BaHomeGreen,
        secondaryContainer = BaHomeSkySurface,
        onSecondaryContainer = BaHomeInk,
        onBackground = BaHomeInk,
        onSurface = BaHomeInk,
        onSurfaceVariant = BaHomeTextMuted,
        error = BaHomeDanger,
        onError = BaHomeOnAction
    )
    MaterialTheme(
        colorScheme = colors,
        typography = BlindAssistTypography,
        shapes = BlindAssistShapes,
        content = content
    )
}

internal val BaNight = Color(0xFF091014)
internal val BaPanel = Color(0xFF101B20)
internal val BaPanelSoft = Color(0xFF1A2B32)
internal val BaPanelRaised = Color(0xFF20343B)
internal val BaMint = Color(0xFF8BE3BA)
internal val BaSky = Color(0xFF8AC7FF)
internal val BaAmber = Color(0xFFFFD66B)
internal val BaDanger = Color(0xFFFF6B7E)
internal val BaText = Color(0xFFF4EFE6)
internal val BaTextMuted = Color(0xFFAFC0C6)
internal val BaInk = Color(0xFF0B1114)
internal val BaHairline = Color(0xFF34464D)
internal val BaHairlineSoft = Color(0xFF223239)
internal val BaMintWash = Color(0xFF17372F)
internal val BaSkyWash = Color(0xFF172D3D)

// Home shell palette derived from the selected warm, spatial light design.
// These tokens stay separate from the dark camera/control palette above.
internal val BaHomeBackground = Color(0xFFF7F6F1)
internal val BaHomeSurface = Color(0xFFFFFDF9)
internal val BaHomeControlRail = Color(0xFFF0F2ED)
internal val BaHomeInk = Color(0xFF091A32)
internal val BaHomeGreen = Color(0xFF197B55)
internal val BaHomeCobalt = Color(0xFF2267B8)
internal val BaHomeHairline = Color(0xFFDEDED7)
internal val BaHomeActionStart = Color(0xFF173E31)
internal val BaHomeActionEnd = Color(0xFF0F2F26)
internal val BaHomeActionTextMuted = Color(0xFFBFD8CC)
internal val BaHomeCoolWash = Color(0xFFF3F7F8)
internal val BaHomeWarmWash = Color(0xFFFAF6EE)
internal val BaHomeSageWash = Color(0x38D9E9DD)
internal val BaHomeBlueWash = Color(0x2FD9E8F5)
internal val BaHomeNavInactive = Color(0xFF747B80)
internal val BaHomeNavIndicator = Color(0xFFE5EFE7)
internal val BaHomeTextMuted = Color(0xFF637078)
internal val BaHomeSurfaceRaised = Color(0xFFEDF1ED)
internal val BaHomeSkySurface = Color(0xFFE9F1F8)
internal val BaHomeAmberSurface = Color(0xFFF8EFD9)
internal val BaHomeAmber = Color(0xFF806019)
internal val BaHomeDanger = Color(0xFFB3261E)
internal val BaHomeOnAction = Color.White

internal val BaShapeCompact = RoundedCornerShape(14.dp)
internal val BaShapeControl = RoundedCornerShape(18.dp)
internal val BaShapeCard = RoundedCornerShape(24.dp)
internal val BaShapeHero = RoundedCornerShape(28.dp)

private val BlindAssistShapes = Shapes(
    extraSmall = RoundedCornerShape(10.dp),
    small = BaShapeCompact,
    medium = BaShapeControl,
    large = BaShapeCard,
    extraLarge = RoundedCornerShape(30.dp)
)

private val BlindAssistTypography = Typography(
    displaySmall = androidx.compose.ui.text.TextStyle(
        fontSize = 40.sp,
        lineHeight = 47.sp,
        fontWeight = FontWeight.Bold,
        letterSpacing = (-0.7).sp
    ),
    headlineLarge = androidx.compose.ui.text.TextStyle(
        fontSize = 35.sp,
        lineHeight = 42.sp,
        fontWeight = FontWeight.Bold,
        letterSpacing = (-0.5).sp
    ),
    headlineMedium = androidx.compose.ui.text.TextStyle(
        fontSize = 28.sp,
        lineHeight = 34.sp,
        fontWeight = FontWeight.Bold,
        letterSpacing = (-0.25).sp
    ),
    headlineSmall = androidx.compose.ui.text.TextStyle(
        fontSize = 22.sp,
        lineHeight = 28.sp,
        fontWeight = FontWeight.SemiBold
    ),
    titleLarge = androidx.compose.ui.text.TextStyle(
        fontSize = 20.sp,
        lineHeight = 26.sp,
        fontWeight = FontWeight.SemiBold
    ),
    titleMedium = androidx.compose.ui.text.TextStyle(
        fontSize = 17.sp,
        lineHeight = 23.sp,
        fontWeight = FontWeight.SemiBold
    ),
    titleSmall = androidx.compose.ui.text.TextStyle(
        fontSize = 15.sp,
        lineHeight = 21.sp,
        fontWeight = FontWeight.SemiBold
    ),
    bodyLarge = androidx.compose.ui.text.TextStyle(
        fontSize = 17.sp,
        lineHeight = 26.sp,
        fontWeight = FontWeight.Normal
    ),
    bodyMedium = androidx.compose.ui.text.TextStyle(
        fontSize = 15.sp,
        lineHeight = 22.sp,
        fontWeight = FontWeight.Normal
    ),
    bodySmall = androidx.compose.ui.text.TextStyle(
        fontSize = 13.sp,
        lineHeight = 19.sp,
        fontWeight = FontWeight.Normal
    ),
    labelLarge = androidx.compose.ui.text.TextStyle(
        fontSize = 14.sp,
        lineHeight = 20.sp,
        fontWeight = FontWeight.SemiBold
    ),
    labelMedium = androidx.compose.ui.text.TextStyle(
        fontSize = 12.sp,
        lineHeight = 16.sp,
        fontWeight = FontWeight.SemiBold,
        letterSpacing = 0.2.sp
    ),
    labelSmall = androidx.compose.ui.text.TextStyle(
        fontSize = 11.sp,
        lineHeight = 15.sp,
        fontWeight = FontWeight.SemiBold,
        letterSpacing = 0.25.sp
    )
)
