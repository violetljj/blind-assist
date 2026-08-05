#include <Arduino.h>
#include <ESPmDNS.h>
#include <Preferences.h>
#include <VL53L1X.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <esp_camera.h>
#include <esp_heap_caps.h>
#include <esp_http_server.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <inttypes.h>
#include <lwip/sockets.h>
#include <lwip/tcp.h>

#include "web_ui.h"

namespace {

constexpr bool kEnableTofSampling = true;
constexpr bool kEnableCameraPsramDma = false;
constexpr bool kEnableStreamTcpNoDelay = true;
constexpr bool kCoalesceStreamPreamble = false;
constexpr bool kReuseStreamFrameCopyBuffer = false;
constexpr BaseType_t kStreamServerCoreId = tskNO_AFFINITY;
constexpr unsigned kStreamServerTaskPriority = tskIDLE_PRIORITY + 5;
constexpr const char* kFirmwareVersion =
    kEnableCameraPsramDma ? "atoms3r_m12_tof4m_slow_frame_r6_psram_dma"
    : kReuseStreamFrameCopyBuffer
        ? "atoms3r_m12_tof4m_stream_r11_reuse_copy_buffer"
        : "atoms3r_m12_tof4m_stream_r11_per_frame_copy_buffer";
constexpr char kSampleSchema[] = "blindassist_atoms3r_tof4m_sample_r0";
constexpr char kEventSchema[] = "blindassist_atoms3r_tof4m_event_r0";
constexpr char kSensorId[] = "m5stack_unit_tof4m_vl53l1x";
constexpr uint8_t kSdaPin = 2;
constexpr uint8_t kSclPin = 1;
constexpr uint8_t kTofAddress = 0x29;
constexpr uint32_t kI2cFrequencyHz = 400000;
constexpr uint32_t kSerialBaud = 115200;
constexpr uint32_t kTimingBudgetUs = 50000;
constexpr uint32_t kInterMeasurementMs = 50;
constexpr uint16_t kSensorTimeoutMs = 500;
constexpr uint16_t kMinimumAdmittedRangeMm = 40;
constexpr uint16_t kMaximumAdmittedRangeMm = 4000;
constexpr char kPreferencesNamespace[] = "tof4m_wifi";
constexpr char kSetupSsid[] = "AtomS3R-ToF-Setup";
constexpr char kSetupPassword[] = "blindassist";
constexpr char kMdnsHostname[] = "atoms3r-tof";
constexpr uint32_t kWifiConnectTimeoutMs = 20000;
constexpr uint32_t kWifiReconnectIntervalMs = 5000;
constexpr uint32_t kCameraLockTimeoutMs = 2000;
constexpr uint16_t kTimingUdpPort = 3333;

constexpr int kCameraPowerPin = 18;
constexpr int kCameraXclkPin = 21;
constexpr int kCameraSiodPin = 12;
constexpr int kCameraSiocPin = 9;
constexpr int kCameraY9Pin = 13;
constexpr int kCameraY8Pin = 11;
constexpr int kCameraY7Pin = 17;
constexpr int kCameraY6Pin = 4;
constexpr int kCameraY5Pin = 48;
constexpr int kCameraY4Pin = 46;
constexpr int kCameraY3Pin = 42;
constexpr int kCameraY2Pin = 3;
constexpr int kCameraVsyncPin = 10;
constexpr int kCameraHrefPin = 14;
constexpr int kCameraPclkPin = 40;

VL53L1X sensor;
TwoWire tof_bus(0);
bool sensor_ready = false;
uint64_t sample_index = 0;
uint64_t next_retry_us = 0;
uint64_t next_health_event_us = 0;
char sequence_id[40] = {};
char clock_domain[64] = {};
httpd_handle_t control_httpd = nullptr;
httpd_handle_t stream_httpd = nullptr;
bool setup_mode = false;
bool camera_ready = false;
bool camera_psram_dma_enabled = false;
char camera_failure_status[48] = "NOT_ATTEMPTED";
SemaphoreHandle_t camera_mutex = nullptr;
uint64_t next_wifi_reconnect_us = 0;
uint32_t wifi_reconnect_attempts = 0;
bool wifi_was_connected = false;
WiFiUDP timing_udp;
bool timing_udp_ready = false;
TaskHandle_t timing_udp_task_handle = nullptr;

struct TimingRequest {
  char magic[4];
  uint32_t request_id;
  uint64_t host_send_monotonic_ns;
};

struct TimingResponse {
  char magic[4];
  uint32_t request_id;
  uint64_t device_receive_us;
  uint64_t device_send_us;
};

static_assert(sizeof(TimingRequest) == 16);
static_assert(sizeof(TimingResponse) == 24);

struct FrameSizeOption {
  const char* name;
  framesize_t value;
  uint16_t width;
  uint16_t height;
};

constexpr FrameSizeOption kFrameSizeOptions[] = {
    {"VGA", FRAMESIZE_VGA, 640, 480},
    {"SVGA", FRAMESIZE_SVGA, 800, 600},
    {"XGA", FRAMESIZE_XGA, 1024, 768},
    {"SXGA", FRAMESIZE_SXGA, 1280, 1024},
    {"UXGA", FRAMESIZE_UXGA, 1600, 1200},
};

struct CameraRuntimeSettings {
  const FrameSizeOption* frame_size;
  uint8_t jpeg_quality;
  int8_t brightness;
  bool auto_exposure;
  int8_t exposure_compensation;
  uint16_t manual_exposure;
};

CameraRuntimeSettings camera_settings = {&kFrameSizeOptions[2], 10, 1, true,
                                         0, 300};

struct FrameStats {
  uint64_t total_frames;
  uint64_t window_started_us;
  uint64_t last_frame_us;
  uint32_t window_frames;
  float recent_fps;
  uint16_t stream_clients;
};

FrameStats frame_stats = {0, 0, 0, 0, 0.0F, 0};
portMUX_TYPE frame_stats_mux = portMUX_INITIALIZER_UNLOCKED;

struct SharedRange {
  bool valid;
  uint16_t range_mm;
  uint8_t range_status_code;
  uint64_t timestamp_ns;
  const char* status;
};

SharedRange shared_range = {false, 0, 0, 0, "NOT_READY"};
constexpr size_t kRangeHistoryCapacity = 32;
SharedRange range_history[kRangeHistoryCapacity] = {};
size_t range_history_count = 0;
size_t range_history_next = 0;
uint64_t range_update_count = 0;
portMUX_TYPE range_mux = portMUX_INITIALIZER_UNLOCKED;
uint64_t next_frame_sequence = 0;
portMUX_TYPE frame_sequence_mux = portMUX_INITIALIZER_UNLOCKED;

void emitEvent(const char* event, const char* status);
void timingUdpTask(void* context);

constexpr char kStreamContentType[] =
    "multipart/x-mixed-replace;boundary=123456789000000000000987654321";
constexpr char kStreamBoundary[] =
    "\r\n--123456789000000000000987654321\r\n";

uint64_t monotonicNs() {
  return static_cast<uint64_t>(esp_timer_get_time()) * 1000ULL;
}

uint64_t monotonicUs() {
  return static_cast<uint64_t>(esp_timer_get_time());
}

uint64_t cameraTimestampUs(const camera_fb_t* frame) {
  return static_cast<uint64_t>(frame->timestamp.tv_sec) * 1000000ULL +
         static_cast<uint64_t>(frame->timestamp.tv_usec);
}

uint64_t claimFrameSequence() {
  portENTER_CRITICAL(&frame_sequence_mux);
  const uint64_t sequence = next_frame_sequence++;
  portEXIT_CRITICAL(&frame_sequence_mux);
  return sequence;
}

void startTimingUdp() {
  if (!timing_udp_ready) {
    timing_udp_ready = timing_udp.begin(kTimingUdpPort) == 1;
    if (timing_udp_ready && timing_udp_task_handle == nullptr) {
      const BaseType_t created = xTaskCreatePinnedToCore(
          timingUdpTask, "timing_udp", 4096, nullptr, 2,
          &timing_udp_task_handle, 1);
      if (created != pdPASS) {
        timing_udp_ready = false;
        timing_udp_task_handle = nullptr;
      }
    }
    emitEvent("timing_udp", timing_udp_ready ? "READY_PORT_3333"
                                              : "START_FAILED");
  }
}

void processTimingUdp() {
  if (!timing_udp_ready) {
    return;
  }
  const int packet_size = timing_udp.parsePacket();
  if (packet_size <= 0) {
    return;
  }
  const uint64_t received_us = monotonicUs();
  TimingRequest request = {};
  const int bytes_read = timing_udp.read(
      reinterpret_cast<uint8_t*>(&request), sizeof(request));
  while (timing_udp.available() > 0) {
    timing_udp.read();
  }
  if (packet_size != sizeof(request) || bytes_read != sizeof(request) ||
      memcmp(request.magic, "BAT0", 4) != 0) {
    return;
  }
  TimingResponse response = {};
  memcpy(response.magic, "BAT1", 4);
  response.request_id = request.request_id;
  response.device_receive_us = received_us;
  timing_udp.beginPacket(timing_udp.remoteIP(), timing_udp.remotePort());
  response.device_send_us = monotonicUs();
  timing_udp.write(reinterpret_cast<const uint8_t*>(&response),
                   sizeof(response));
  timing_udp.endPacket();
}

void timingUdpTask(void* context) {
  (void)context;
  for (;;) {
    processTimingUdp();
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

SharedRange snapshotRange() {
  SharedRange snapshot;
  portENTER_CRITICAL(&range_mux);
  snapshot = shared_range;
  portEXIT_CRITICAL(&range_mux);
  return snapshot;
}

uint64_t snapshotRangeUpdateCount() {
  portENTER_CRITICAL(&range_mux);
  const uint64_t snapshot = range_update_count;
  portEXIT_CRITICAL(&range_mux);
  return snapshot;
}

SharedRange snapshotNearestRange(uint64_t frame_timestamp_us) {
  SharedRange nearest = snapshotRange();
  uint64_t best_delta_us = UINT64_MAX;
  portENTER_CRITICAL(&range_mux);
  for (size_t index = 0; index < range_history_count; ++index) {
    const SharedRange candidate = range_history[index];
    const uint64_t candidate_us = candidate.timestamp_ns / 1000ULL;
    const uint64_t delta_us = candidate_us >= frame_timestamp_us
                                  ? candidate_us - frame_timestamp_us
                                  : frame_timestamp_us - candidate_us;
    if (delta_us < best_delta_us) {
      nearest = candidate;
      best_delta_us = delta_us;
    }
  }
  portEXIT_CRITICAL(&range_mux);
  return nearest;
}

void updateFrameStats(bool client_delta, bool client_connected,
                      bool frame_sent) {
  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  portENTER_CRITICAL(&frame_stats_mux);
  if (client_delta) {
    if (client_connected) {
      ++frame_stats.stream_clients;
    } else if (frame_stats.stream_clients > 0) {
      --frame_stats.stream_clients;
    }
  }
  if (frame_sent) {
    if (frame_stats.window_started_us == 0) {
      frame_stats.window_started_us = now_us;
    }
    ++frame_stats.total_frames;
    ++frame_stats.window_frames;
    frame_stats.last_frame_us = now_us;
    const uint64_t elapsed_us = now_us - frame_stats.window_started_us;
    if (elapsed_us >= 1000000ULL) {
      frame_stats.recent_fps =
          static_cast<float>(frame_stats.window_frames) * 1000000.0F /
          static_cast<float>(elapsed_us);
      frame_stats.window_frames = 0;
      frame_stats.window_started_us = now_us;
    }
  }
  portEXIT_CRITICAL(&frame_stats_mux);
}

FrameStats snapshotFrameStats() {
  FrameStats snapshot;
  portENTER_CRITICAL(&frame_stats_mux);
  snapshot = frame_stats;
  portEXIT_CRITICAL(&frame_stats_mux);
  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  if (snapshot.last_frame_us == 0 || now_us - snapshot.last_frame_us > 2000000ULL) {
    snapshot.recent_fps = 0.0F;
  }
  return snapshot;
}

const FrameSizeOption* findFrameSize(const char* name) {
  for (const FrameSizeOption& option : kFrameSizeOptions) {
    if (strcmp(option.name, name) == 0) {
      return &option;
    }
  }
  return nullptr;
}

const FrameSizeOption* findFrameSize(uint16_t width, uint16_t height) {
  for (const FrameSizeOption& option : kFrameSizeOptions) {
    if (option.width == width && option.height == height) {
      return &option;
    }
  }
  return nullptr;
}

bool parseInteger(const char* value, long minimum, long maximum,
                  long* parsed) {
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  char* end = nullptr;
  const long candidate = strtol(value, &end, 10);
  if (end == value || *end != '\0' || candidate < minimum ||
      candidate > maximum) {
    return false;
  }
  *parsed = candidate;
  return true;
}

bool readRequestBody(httpd_req_t* request, char* body, size_t body_size) {
  if (request->content_len <= 0 ||
      static_cast<size_t>(request->content_len) >= body_size) {
    return false;
  }
  int received = 0;
  while (received < request->content_len) {
    const int result = httpd_req_recv(request, body + received,
                                      request->content_len - received);
    if (result <= 0) {
      return false;
    }
    received += result;
  }
  body[received] = '\0';
  return true;
}

bool probeTofAddress() {
  tof_bus.beginTransmission(kTofAddress);
  return tof_bus.endTransmission() == 0;
}

void emitEvent(const char* event, const char* status) {
  Serial.printf(
      "{\"schema\":\"%s\",\"firmware_version\":\"%s\","
      "\"sequence_id\":\"%s\",\"timestamp_ns\":%" PRIu64 ","
      "\"clock_domain\":\"%s\",\"event\":\"%s\","
      "\"status\":\"%s\"}\n",
      kEventSchema, kFirmwareVersion, sequence_id, monotonicNs(), clock_domain,
      event, status);
}

void updateSharedRange(bool valid, uint16_t range_mm, uint8_t range_status_code,
                       uint64_t timestamp_ns, const char* status) {
  portENTER_CRITICAL(&range_mux);
  shared_range.valid = valid;
  shared_range.range_mm = range_mm;
  shared_range.range_status_code = range_status_code;
  shared_range.timestamp_ns = timestamp_ns;
  shared_range.status = status;
  range_history[range_history_next] = shared_range;
  range_history_next = (range_history_next + 1) % kRangeHistoryCapacity;
  if (range_history_count < kRangeHistoryCapacity) {
    ++range_history_count;
  }
  ++range_update_count;
  portEXIT_CRITICAL(&range_mux);
}

bool initializeSensor() {
  if (!probeTofAddress()) {
    emitEvent("i2c_probe", "TOF4M_0X29_NOT_FOUND");
    return false;
  }
  emitEvent("i2c_probe", "TOF4M_0X29_FOUND");

  sensor.setBus(&tof_bus);
  sensor.setTimeout(kSensorTimeoutMs);
  if (!sensor.init()) {
    emitEvent("sensor_init", "VL53L1X_INIT_FAILED");
    return false;
  }
  if (!sensor.setDistanceMode(VL53L1X::Long)) {
    emitEvent("sensor_config", "LONG_DISTANCE_MODE_FAILED");
    return false;
  }
  if (!sensor.setMeasurementTimingBudget(kTimingBudgetUs)) {
    emitEvent("sensor_config", "TIMING_BUDGET_FAILED");
    return false;
  }

  sensor.setROISize(16, 16);
  sensor.startContinuous(kInterMeasurementMs);
  emitEvent("sensor_init", "READY");
  return true;
}

const char* classifyMeasurement(bool timed_out, VL53L1X::RangeStatus range_status,
                                uint16_t range_mm) {
  if (timed_out) {
    return "INVALID_TIMEOUT";
  }
  if (range_status != VL53L1X::RangeValid) {
    return "INVALID_SENSOR_STATUS";
  }
  if (range_mm < kMinimumAdmittedRangeMm ||
      range_mm > kMaximumAdmittedRangeMm) {
    return "INVALID_RANGE";
  }
  return "VALID";
}

void emitSample() {
  sensor.read(true);
  const uint64_t timestamp_ns = monotonicNs();
  const bool timed_out = sensor.timeoutOccurred();
  const uint16_t range_mm = sensor.ranging_data.range_mm;
  const VL53L1X::RangeStatus range_status = sensor.ranging_data.range_status;
  const char* measurement_status =
      classifyMeasurement(timed_out, range_status, range_mm);
  const bool valid = strcmp(measurement_status, "VALID") == 0;
  updateSharedRange(valid, range_mm, static_cast<uint8_t>(range_status),
                    timestamp_ns, measurement_status);

  Serial.printf(
      "{\"schema\":\"%s\",\"firmware_version\":\"%s\","
      "\"sequence_id\":\"%s\",\"sample_index\":%" PRIu64 ","
      "\"timestamp_ns\":%" PRIu64 ",\"timestamp_semantics\":"
      "\"sensor_read_complete\",\"clock_domain\":\"%s\","
      "\"sensor_id\":\"%s\",\"i2c_address_7bit\":%u,"
      "\"sda_gpio\":%u,\"scl_gpio\":%u,\"distance_mode\":\"LONG\","
      "\"timing_budget_us\":%lu,\"inter_measurement_ms\":%lu,"
      "\"roi_width_spad\":16,\"roi_height_spad\":16,"
      "\"measurement_status\":\"%s\",\"timeout\":%s,"
      "\"range_status_code\":%u,\"range_mm\":%u,\"range_m\":",
      kSampleSchema, kFirmwareVersion, sequence_id, sample_index++, timestamp_ns,
      clock_domain, kSensorId, kTofAddress, kSdaPin, kSclPin,
      static_cast<unsigned long>(kTimingBudgetUs),
      static_cast<unsigned long>(kInterMeasurementMs), measurement_status,
      timed_out ? "true" : "false", static_cast<unsigned>(range_status),
      range_mm);

  if (valid) {
    Serial.printf("%.3f", static_cast<double>(range_mm) / 1000.0);
  } else {
    Serial.print("null");
  }
  Serial.printf(
      ",\"peak_signal_rate_mcps\":%.6f,\"ambient_rate_mcps\":%.6f}\n",
      static_cast<double>(sensor.ranging_data.peak_signal_count_rate_MCPS),
      static_cast<double>(sensor.ranging_data.ambient_count_rate_MCPS));
}

bool initializeCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = kCameraY2Pin;
  config.pin_d1 = kCameraY3Pin;
  config.pin_d2 = kCameraY4Pin;
  config.pin_d3 = kCameraY5Pin;
  config.pin_d4 = kCameraY6Pin;
  config.pin_d5 = kCameraY7Pin;
  config.pin_d6 = kCameraY8Pin;
  config.pin_d7 = kCameraY9Pin;
  config.pin_xclk = kCameraXclkPin;
  config.pin_pclk = kCameraPclkPin;
  config.pin_vsync = kCameraVsyncPin;
  config.pin_href = kCameraHrefPin;
  config.pin_sccb_sda = kCameraSiodPin;
  config.pin_sccb_scl = kCameraSiocPin;
  config.pin_pwdn = -1;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.frame_size = camera_settings.frame_size->value;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = camera_settings.jpeg_quality;
  config.fb_count = psramFound() ? 2 : 1;
  if (!psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  const esp_err_t error = esp_camera_init(&config);
  if (error != ESP_OK) {
    snprintf(camera_failure_status, sizeof(camera_failure_status),
             "OV3660_INIT_FAILED_0X%X", static_cast<unsigned>(error));
    emitEvent("camera_init", camera_failure_status);
    return false;
  }

  if (kEnableCameraPsramDma) {
    const esp_err_t psram_dma_error = esp_camera_set_psram_mode(true);
    if (psram_dma_error != ESP_OK) {
      snprintf(camera_failure_status, sizeof(camera_failure_status),
               "PSRAM_DMA_FAILED_0X%X",
               static_cast<unsigned>(psram_dma_error));
      emitEvent("camera_init", camera_failure_status);
      esp_camera_deinit();
      return false;
    }
  }
  camera_psram_dma_enabled = esp_camera_get_psram_mode();

  sensor_t* camera_sensor = esp_camera_sensor_get();
  if (camera_sensor != nullptr && camera_sensor->id.PID == OV3660_PID) {
    camera_sensor->set_vflip(camera_sensor, 1);
    camera_sensor->set_brightness(camera_sensor, camera_settings.brightness);
    camera_sensor->set_saturation(camera_sensor, -2);
    camera_sensor->set_exposure_ctrl(camera_sensor, 1);
    camera_sensor->set_ae_level(
        camera_sensor, camera_settings.exposure_compensation);
  }
  emitEvent("camera_init", "READY_XGA_JPEG_Q10_CONTROLS_R2");
  return true;
}

String urlDecode(const char* value) {
  String decoded;
  const size_t value_length = strlen(value);
  decoded.reserve(value_length);
  for (size_t index = 0; index < value_length; ++index) {
    if (value[index] == '+') {
      decoded += ' ';
      continue;
    }
    if (value[index] == '%' && index + 2 < value_length &&
        isxdigit(value[index + 1]) &&
        isxdigit(value[index + 2])) {
      char hex[3] = {value[index + 1], value[index + 2], '\0'};
      decoded += static_cast<char>(strtol(hex, nullptr, 16));
      index += 2;
      continue;
    }
    decoded += value[index];
  }
  return decoded;
}

esp_err_t sendHtml(httpd_req_t* request, const char* html) {
  httpd_resp_set_type(request, "text/html; charset=utf-8");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  return httpd_resp_send(request, html, HTTPD_RESP_USE_STRLEN);
}

esp_err_t rootHandler(httpd_req_t* request) {
  return sendHtml(request, setup_mode ? kSetupHtml : kDashboardHtml);
}

esp_err_t statusPageHandler(httpd_req_t* request) {
  return sendHtml(request, kStatusHtml);
}

esp_err_t sendJson(httpd_req_t* request, const char* json) {
  httpd_resp_set_type(request, "application/json");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  return httpd_resp_send(request, json, HTTPD_RESP_USE_STRLEN);
}

esp_err_t sendServiceUnavailable(httpd_req_t* request, const char* message) {
  httpd_resp_set_status(request, "503 Service Unavailable");
  httpd_resp_set_type(request, "text/plain; charset=utf-8");
  return httpd_resp_send(request, message, HTTPD_RESP_USE_STRLEN);
}

void formatCameraSettingsJson(char* body, size_t body_size) {
  snprintf(body, body_size,
           "{\"resolution\":\"%s\",\"width\":%u,\"height\":%u,"
           "\"jpeg_quality\":%u,\"brightness\":%d,"
           "\"auto_exposure\":%s,\"exposure_compensation\":%d,"
           "\"manual_exposure\":%u}",
           camera_settings.frame_size->name, camera_settings.frame_size->width,
           camera_settings.frame_size->height, camera_settings.jpeg_quality,
           camera_settings.brightness,
           camera_settings.auto_exposure ? "true" : "false",
           camera_settings.exposure_compensation,
           camera_settings.manual_exposure);
}

esp_err_t cameraSettingsGetHandler(httpd_req_t* request) {
  char body[320];
  formatCameraSettingsJson(body, sizeof(body));
  return sendJson(request, body);
}

esp_err_t cameraSettingsPostHandler(httpd_req_t* request) {
  if (!camera_ready || camera_mutex == nullptr) {
    return sendServiceUnavailable(request, "camera not ready");
  }
  char body[321] = {};
  if (!readRequestBody(request, body, sizeof(body))) {
    return httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST,
                               "invalid camera settings body");
  }

  char resolution[12] = {};
  char quality_text[8] = {};
  char brightness_text[8] = {};
  char auto_exposure_text[8] = {};
  char compensation_text[8] = {};
  char manual_exposure_text[8] = {};
  if (httpd_query_key_value(body, "resolution", resolution,
                            sizeof(resolution)) != ESP_OK ||
      httpd_query_key_value(body, "quality", quality_text,
                            sizeof(quality_text)) != ESP_OK ||
      httpd_query_key_value(body, "brightness", brightness_text,
                            sizeof(brightness_text)) != ESP_OK ||
      httpd_query_key_value(body, "auto_exposure", auto_exposure_text,
                            sizeof(auto_exposure_text)) != ESP_OK ||
      httpd_query_key_value(body, "exposure_compensation", compensation_text,
                            sizeof(compensation_text)) != ESP_OK ||
      httpd_query_key_value(body, "manual_exposure", manual_exposure_text,
                            sizeof(manual_exposure_text)) != ESP_OK) {
    return httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST,
                               "missing camera setting");
  }

  const FrameSizeOption* frame_size = findFrameSize(resolution);
  long quality = 0;
  long brightness = 0;
  long auto_exposure = 0;
  long compensation = 0;
  long manual_exposure = 0;
  if (frame_size == nullptr ||
      !parseInteger(quality_text, 6, 30, &quality) ||
      !parseInteger(brightness_text, -2, 2, &brightness) ||
      !parseInteger(auto_exposure_text, 0, 1, &auto_exposure) ||
      !parseInteger(compensation_text, -2, 2, &compensation) ||
      !parseInteger(manual_exposure_text, 0, 1200, &manual_exposure)) {
    return httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST,
                               "camera setting out of range");
  }

  if (xSemaphoreTake(camera_mutex, pdMS_TO_TICKS(kCameraLockTimeoutMs)) !=
      pdTRUE) {
    return sendServiceUnavailable(request, "camera busy; retry");
  }
  sensor_t* camera_sensor = esp_camera_sensor_get();
  int result = camera_sensor == nullptr ? -1 : 0;
  if (camera_sensor != nullptr) {
    result |= camera_sensor->set_framesize(camera_sensor, frame_size->value);
    result |= camera_sensor->set_quality(camera_sensor, quality);
    result |= camera_sensor->set_brightness(camera_sensor, brightness);
    result |= camera_sensor->set_exposure_ctrl(camera_sensor, auto_exposure);
    if (auto_exposure != 0) {
      result |= camera_sensor->set_ae_level(camera_sensor, compensation);
    } else {
      result |= camera_sensor->set_aec_value(camera_sensor, manual_exposure);
    }
    if (result == 0) {
      for (uint8_t index = 0; index < 3; ++index) {
        camera_fb_t* transition_frame = esp_camera_fb_get();
        if (transition_frame == nullptr) {
          result = -1;
          break;
        }
        esp_camera_fb_return(transition_frame);
        delay(20);
      }
    }
  }
  if (result == 0) {
    camera_settings.frame_size = frame_size;
    camera_settings.jpeg_quality = static_cast<uint8_t>(quality);
    camera_settings.brightness = static_cast<int8_t>(brightness);
    camera_settings.auto_exposure = auto_exposure != 0;
    camera_settings.exposure_compensation =
        static_cast<int8_t>(compensation);
    camera_settings.manual_exposure =
        static_cast<uint16_t>(manual_exposure);
  }
  xSemaphoreGive(camera_mutex);
  if (result != 0) {
    emitEvent("camera_settings", "APPLY_FAILED");
    return httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR,
                               "camera rejected setting");
  }
  emitEvent("camera_settings", "APPLIED_SESSION_ONLY");
  char response[320];
  formatCameraSettingsJson(response, sizeof(response));
  return sendJson(request, response);
}

esp_err_t rangeHandler(httpd_req_t* request) {
  const SharedRange snapshot = snapshotRange();

  const uint64_t now_ns = monotonicNs();
  const uint64_t age_ms = snapshot.timestamp_ns == 0
                              ? 0
                              : (now_ns - snapshot.timestamp_ns) / 1000000ULL;
  char body[256];
  if (snapshot.valid) {
    snprintf(body, sizeof(body),
             "{\"valid\":true,\"range_m\":%.3f,\"range_mm\":%u,"
             "\"status\":\"VALID\",\"range_status_code\":%u,"
             "\"age_ms\":%" PRIu64 "}",
             static_cast<double>(snapshot.range_mm) / 1000.0,
             snapshot.range_mm, snapshot.range_status_code, age_ms);
  } else {
    snprintf(body, sizeof(body),
             "{\"valid\":false,\"range_m\":null,\"range_mm\":%u,"
             "\"status\":\"%s\",\"range_status_code\":%u,"
             "\"age_ms\":%" PRIu64 "}",
             snapshot.range_mm, snapshot.status, snapshot.range_status_code,
             age_ms);
  }
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");
  return sendJson(request, body);
}

esp_err_t statusHandler(httpd_req_t* request) {
  const SharedRange range = snapshotRange();
  const FrameStats frames = snapshotFrameStats();
  const uint64_t now_ns = monotonicNs();
  const uint64_t range_age_ms =
      range.timestamp_ns == 0 ? 0 : (now_ns - range.timestamp_ns) / 1000000ULL;
  const bool wifi_connected = WiFi.status() == WL_CONNECTED;
  const String ip = wifi_connected ? WiFi.localIP().toString() : String("");
  char body[1024];
  snprintf(
      body, sizeof(body),
      "{\"firmware_version\":\"%s\",\"sequence_id\":\"%s\","
      "\"uptime_ms\":%" PRIu64 ",\"free_heap_bytes\":%u,"
      "\"chip_temperature_c\":%.2f,"
      "\"chip_temperature_semantics\":\"esp32_internal_sensor_not_ambient\","
      "\"wifi\":{\"connected\":%s,\"status_code\":%d,"
      "\"ip\":\"%s\",\"rssi_dbm\":%d,\"reconnect_attempts\":%u},"
      "\"camera\":{\"ready\":%s,\"resolution\":\"%s\","
      "\"width\":%u,\"height\":%u,\"jpeg_quality\":%u,"
      "\"brightness\":%d,\"auto_exposure\":%s,"
      "\"exposure_compensation\":%d,\"manual_exposure\":%u,"
      "\"psram_dma_enabled\":%s,\"frame_buffer_count\":2,"
      "\"grab_mode\":\"LATEST\",\"stream_tcp_nodelay_configured\":%s,"
      "\"stream_preamble_coalesced_configured\":%s,"
      "\"stream_frame_copy_buffer_reused_configured\":%s,"
      "\"stream_server_core_configured\":%d,"
      "\"stream_server_task_priority\":%u,"
      "\"recent_fps\":%.2f,\"total_frames\":%" PRIu64 ","
      "\"stream_clients\":%u},"
      "\"tof\":{\"ready\":%s,\"sampling_enabled\":%s,\"valid\":%s,"
      "\"status\":\"%s\","
      "\"range_mm\":%u,\"range_status_code\":%u,\"age_ms\":%" PRIu64
      "}}",
      kFirmwareVersion, sequence_id, now_ns / 1000000ULL, ESP.getFreeHeap(),
      static_cast<double>(temperatureRead()),
      wifi_connected ? "true" : "false", static_cast<int>(WiFi.status()),
      ip.c_str(), wifi_connected ? WiFi.RSSI() : 0, wifi_reconnect_attempts,
      camera_ready ? "true" : "false", camera_settings.frame_size->name,
      camera_settings.frame_size->width, camera_settings.frame_size->height,
      camera_settings.jpeg_quality, camera_settings.brightness,
      camera_settings.auto_exposure ? "true" : "false",
      camera_settings.exposure_compensation, camera_settings.manual_exposure,
      camera_psram_dma_enabled ? "true" : "false",
      kEnableStreamTcpNoDelay ? "true" : "false",
      kCoalesceStreamPreamble ? "true" : "false",
      kReuseStreamFrameCopyBuffer ? "true" : "false",
      static_cast<int>(kStreamServerCoreId), kStreamServerTaskPriority,
      static_cast<double>(frames.recent_fps), frames.total_frames,
      frames.stream_clients, sensor_ready ? "true" : "false",
      kEnableTofSampling ? "true" : "false",
      range.valid ? "true" : "false", range.status, range.range_mm,
      range.range_status_code, range_age_ms);
  return sendJson(request, body);
}

esp_err_t timeSyncHandler(httpd_req_t* request) {
  const uint64_t request_received_us = monotonicUs();
  char body[320];
  const uint64_t response_ready_us = monotonicUs();
  snprintf(body, sizeof(body),
           "{\"schema\":\"blindassist_atoms3r_time_sync_r0\","
           "\"sequence_id\":\"%s\",\"clock_domain\":\"%s\","
           "\"device_request_received_us\":%" PRIu64 ","
           "\"device_response_ready_us\":%" PRIu64 "}",
           sequence_id, clock_domain, request_received_us, response_ready_us);
  return sendJson(request, body);
}

esp_err_t snapshotHandler(httpd_req_t* request) {
  if (!camera_ready || camera_mutex == nullptr) {
    return sendServiceUnavailable(request, "camera not ready");
  }
  if (xSemaphoreTake(camera_mutex, pdMS_TO_TICKS(kCameraLockTimeoutMs)) !=
      pdTRUE) {
    return sendServiceUnavailable(request, "camera busy; retry");
  }
  camera_fb_t* frame = esp_camera_fb_get();
  if (frame == nullptr) {
    xSemaphoreGive(camera_mutex);
    return httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR,
                               "snapshot failed");
  }
  const uint64_t frame_sequence = claimFrameSequence();
  const uint64_t capture_timestamp_us = cameraTimestampUs(frame);
  const uint64_t jpeg_ready_timestamp_us = monotonicUs();
  const SharedRange range = snapshotNearestRange(capture_timestamp_us);
  const uint64_t tof_timestamp_us = range.timestamp_ns / 1000ULL;
  const int64_t tof_minus_capture_us =
      static_cast<int64_t>(tof_timestamp_us) -
      static_cast<int64_t>(capture_timestamp_us);
  const FrameSizeOption* actual_frame_size =
      findFrameSize(frame->width, frame->height);
  const uint64_t tof_age_at_jpeg_ready_us =
      tof_timestamp_us == 0 || tof_timestamp_us > jpeg_ready_timestamp_us
          ? 0
          : jpeg_ready_timestamp_us - tof_timestamp_us;
  const uint64_t capture_to_jpeg_ready_us =
      jpeg_ready_timestamp_us >= capture_timestamp_us
          ? jpeg_ready_timestamp_us - capture_timestamp_us
          : 0;
  char metadata[1024];
  snprintf(metadata, sizeof(metadata),
           "{\"schema\":\"blindassist_atoms3r_capture_browser_r1\","
           "\"sequence_id\":\"%s\",\"clock_domain\":\"%s\","
           "\"frame_sequence\":%" PRIu64 ","
           "\"capture_timestamp_us\":%" PRIu64 ","
           "\"capture_timestamp_semantics\":"
           "\"esp32_camera_first_dma_buffer_since_boot\","
           "\"jpeg_ready_timestamp_us\":%" PRIu64 ","
           "\"jpeg_ready_timestamp_semantics\":\"esp_camera_fb_get_return\","
           "\"capture_to_jpeg_ready_us\":%" PRIu64 ","
           "\"tof_timestamp_us\":%" PRIu64 ","
           "\"tof_timestamp_semantics\":\"sensor_read_complete\","
           "\"tof_minus_capture_us\":%" PRId64 ","
           "\"tof_age_at_jpeg_ready_us\":%" PRIu64 ","
           "\"tof_valid\":%s,\"tof_range_mm\":%u,\"tof_status\":\"%s\","
           "\"tof_range_status_code\":%u,\"resolution\":\"%s\","
           "\"width\":%u,\"height\":%u,\"jpeg_quality\":%u}",
           sequence_id, clock_domain, frame_sequence, capture_timestamp_us,
           jpeg_ready_timestamp_us, capture_to_jpeg_ready_us, tof_timestamp_us,
           tof_minus_capture_us, tof_age_at_jpeg_ready_us,
           range.valid ? "true" : "false", range.range_mm, range.status,
           range.range_status_code,
           actual_frame_size == nullptr ? "UNKNOWN" : actual_frame_size->name,
           frame->width, frame->height, camera_settings.jpeg_quality);
  httpd_resp_set_type(request, "image/jpeg");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  httpd_resp_set_hdr(request, "X-Capture-Metadata", metadata);
  const esp_err_t result = httpd_resp_send(
      request, reinterpret_cast<const char*>(frame->buf), frame->len);
  esp_camera_fb_return(frame);
  xSemaphoreGive(camera_mutex);
  return result;
}

esp_err_t saveWifiHandler(httpd_req_t* request) {
  char body[257] = {};
  if (!readRequestBody(request, body, sizeof(body))) {
    return httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST,
                               "invalid form length");
  }

  char encoded_ssid[97] = {};
  char encoded_password[193] = {};
  if (httpd_query_key_value(body, "ssid", encoded_ssid,
                            sizeof(encoded_ssid)) != ESP_OK ||
      httpd_query_key_value(body, "password", encoded_password,
                            sizeof(encoded_password)) != ESP_OK) {
    return httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST,
                               "missing Wi-Fi fields");
  }
  const String ssid = urlDecode(encoded_ssid);
  const String password = urlDecode(encoded_password);
  if (ssid.isEmpty() || ssid.length() > 32 || password.length() > 63) {
    return httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST,
                               "invalid Wi-Fi fields");
  }

  Preferences preferences;
  if (!preferences.begin(kPreferencesNamespace, false)) {
    return httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR,
                               "cannot open local storage");
  }
  const bool ssid_stored = preferences.putString("ssid", ssid) > 0;
  bool password_stored = true;
  if (password.isEmpty()) {
    preferences.remove("password");
  } else {
    password_stored = preferences.putString("password", password) > 0;
  }
  const bool stored = ssid_stored && password_stored;
  preferences.end();
  if (!stored) {
    return httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR,
                               "cannot save Wi-Fi settings");
  }

  sendHtml(request, kSavedHtml);
  delay(1200);
  ESP.restart();
  return ESP_OK;
}

esp_err_t forgetWifiHandler(httpd_req_t* request) {
  Preferences preferences;
  if (preferences.begin(kPreferencesNamespace, false)) {
    preferences.clear();
    preferences.end();
  }
  sendHtml(request, kSavedHtml);
  delay(800);
  ESP.restart();
  return ESP_OK;
}

esp_err_t streamHandler(httpd_req_t* request) {
  httpd_resp_set_type(request, kStreamContentType);
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  updateFrameStats(true, true, false);
  const int socket_fd = httpd_req_to_sockfd(request);
  const int requested_tcp_nodelay = kEnableStreamTcpNoDelay ? 1 : 0;
  if (socket_fd < 0 ||
      setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &requested_tcp_nodelay,
                 sizeof(requested_tcp_nodelay)) != 0) {
    emitEvent("camera_stream", "TCP_NODELAY_APPLY_FAILED");
    updateFrameStats(true, false, false);
    return ESP_FAIL;
  }
  int observed_tcp_nodelay = 0;
  socklen_t observed_tcp_nodelay_length = sizeof(observed_tcp_nodelay);
  if (getsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &observed_tcp_nodelay,
                 &observed_tcp_nodelay_length) != 0) {
    emitEvent("camera_stream", "TCP_NODELAY_READBACK_FAILED");
    updateFrameStats(true, false, false);
    return ESP_FAIL;
  }
  const bool stream_tcp_nodelay_enabled = observed_tcp_nodelay != 0;
  esp_err_t stream_result = ESP_OK;
  uint64_t previous_jpeg_ready_us = 0;
  uint64_t previous_frame_sequence = 0;
  uint64_t previous_response_write_duration_us = 0;
  bool previous_response_write_valid = false;
  uint64_t previous_range_update_count = snapshotRangeUpdateCount();
  uint8_t* reusable_frame_copy = nullptr;
  size_t reusable_frame_copy_capacity = 0;
  while (true) {
    const uint64_t range_updates_at_acquire_start = snapshotRangeUpdateCount();
    const uint64_t frame_acquire_started_us = monotonicUs();
    if (camera_mutex == nullptr ||
        xSemaphoreTake(camera_mutex, portMAX_DELAY) != pdTRUE) {
      stream_result = ESP_FAIL;
      break;
    }
    camera_fb_t* frame = esp_camera_fb_get();
    if (frame == nullptr) {
      xSemaphoreGive(camera_mutex);
      emitEvent("camera_capture", "FRAME_FAILED");
      stream_result = ESP_FAIL;
      break;
    }
    const uint64_t frame_sequence = claimFrameSequence();
    const uint64_t capture_timestamp_us = cameraTimestampUs(frame);
    const uint64_t jpeg_ready_timestamp_us = monotonicUs();
    const uint64_t range_updates_at_jpeg_ready = snapshotRangeUpdateCount();
    const uint64_t tof_updates_during_acquire =
        range_updates_at_jpeg_ready - range_updates_at_acquire_start;
    const uint64_t tof_updates_since_previous_frame =
        range_updates_at_jpeg_ready - previous_range_update_count;
    const uint64_t frame_acquire_duration_us =
        jpeg_ready_timestamp_us - frame_acquire_started_us;
    const uint64_t frame_ready_interval_us =
        previous_jpeg_ready_us == 0
            ? 0
            : jpeg_ready_timestamp_us - previous_jpeg_ready_us;
    const SharedRange range = snapshotNearestRange(capture_timestamp_us);
    const uint64_t tof_timestamp_us = range.timestamp_ns / 1000ULL;
    const int64_t tof_minus_capture_us =
        tof_timestamp_us == 0
            ? 0
            : static_cast<int64_t>(tof_timestamp_us) -
                  static_cast<int64_t>(capture_timestamp_us);
    const size_t frame_length = frame->len;
    const size_t frame_width = frame->width;
    const size_t frame_height = frame->height;
    uint8_t* frame_copy = nullptr;
    if (kReuseStreamFrameCopyBuffer) {
      if (frame_length > reusable_frame_copy_capacity) {
        uint8_t* larger_copy = static_cast<uint8_t*>(heap_caps_realloc(
            reusable_frame_copy, frame_length,
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        if (larger_copy != nullptr) {
          reusable_frame_copy = larger_copy;
          reusable_frame_copy_capacity = frame_length;
        }
      }
      if (frame_length <= reusable_frame_copy_capacity) {
        frame_copy = reusable_frame_copy;
      }
    } else {
      frame_copy = static_cast<uint8_t*>(
          heap_caps_malloc(frame_length, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    }
    if (frame_copy == nullptr) {
      esp_camera_fb_return(frame);
      xSemaphoreGive(camera_mutex);
      emitEvent("camera_stream", "PSRAM_FRAME_COPY_FAILED");
      stream_result = ESP_ERR_NO_MEM;
      break;
    }
    memcpy(frame_copy, frame->buf, frame_length);
    esp_camera_fb_return(frame);
    xSemaphoreGive(camera_mutex);
    sensor_t* camera_sensor = esp_camera_sensor_get();
    const int exposure_value =
        camera_sensor == nullptr ? -1 : camera_sensor->status.aec_value;
    const int wifi_rssi_dbm = WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : 0;
    const uint32_t free_heap_bytes = ESP.getFreeHeap();
    const uint64_t tof_age_at_jpeg_ready_us =
        tof_timestamp_us == 0 || tof_timestamp_us > jpeg_ready_timestamp_us
            ? 0
            : jpeg_ready_timestamp_us - tof_timestamp_us;
    const bool tof_during_acquire =
        tof_timestamp_us >= frame_acquire_started_us &&
        tof_timestamp_us <= jpeg_ready_timestamp_us;
    const uint64_t response_write_started_us = monotonicUs();
    const uint64_t jpeg_metadata_prepare_duration_us =
        response_write_started_us - jpeg_ready_timestamp_us;
    char header[1600];
    const int header_length = snprintf(
        header, sizeof(header),
        "%sContent-Type: image/jpeg\r\nContent-Length: %u\r\n"
        "X-Sequence-Id: %s\r\nX-Clock-Domain: %s\r\n"
        "X-Frame-Sequence: %" PRIu64 "\r\n"
        "X-Capture-Timestamp-Us: %" PRIu64 "\r\n"
        "X-Capture-Timestamp-Semantics: "
        "esp32_camera_first_dma_buffer_since_boot\r\n"
        "X-Jpeg-Ready-Timestamp-Us: %" PRIu64 "\r\n"
        "X-Jpeg-Ready-Timestamp-Semantics: esp_camera_fb_get_return\r\n"
        "X-Device-Send-Start-Timestamp-Us: %" PRIu64 "\r\n"
        "X-Frame-Ready-Interval-Us: %" PRIu64 "\r\n"
        "X-Frame-Acquire-Duration-Us: %" PRIu64 "\r\n"
        "X-Jpeg-Metadata-Prepare-Duration-Us: %" PRIu64 "\r\n"
        "X-Previous-Response-Write-Valid: %s\r\n"
        "X-Previous-Frame-Sequence: %" PRIu64 "\r\n"
        "X-Previous-Response-Write-Duration-Us: %" PRIu64 "\r\n"
        "X-ToF-Timestamp-Us: %" PRIu64 "\r\n"
        "X-ToF-Timestamp-Semantics: sensor_read_complete\r\n"
        "X-ToF-Minus-Capture-Us: %" PRId64 "\r\n"
        "X-ToF-Age-At-Jpeg-Ready-Us: %" PRIu64 "\r\n"
        "X-ToF-During-Acquire: %s\r\n"
        "X-ToF-Updates-During-Acquire: %" PRIu64 "\r\n"
        "X-ToF-Updates-Since-Previous-Frame: %" PRIu64 "\r\n"
        "X-ToF-Sampling-Enabled: %s\r\n"
        "X-ToF-Valid: %s\r\nX-ToF-Range-Mm: %u\r\n"
        "X-ToF-Status: %s\r\nX-ToF-Range-Status-Code: %u\r\n"
        "X-Jpeg-Size-Bytes: %u\r\nX-Width: %u\r\nX-Height: %u\r\n"
        "X-Jpeg-Quality: %u\r\nX-Auto-Exposure: %s\r\n"
        "X-Camera-Psram-Dma-Enabled: %s\r\n"
        "X-Stream-Tcp-Nodelay: %s\r\n"
        "X-Stream-Preamble-Coalesced: %s\r\n"
        "X-Stream-Frame-Copy-Buffer-Reused: %s\r\n"
        "X-Stream-Handler-Core: %d\r\n"
        "X-Stream-Handler-Priority: %u\r\n"
        "X-Exposure-Value: %d\r\nX-Wifi-Rssi-Dbm: %d\r\n"
        "X-Free-Heap-Bytes: %u\r\n\r\n",
        kCoalesceStreamPreamble ? kStreamBoundary : "",
        static_cast<unsigned>(frame_length), sequence_id, clock_domain,
        frame_sequence, capture_timestamp_us, jpeg_ready_timestamp_us,
        response_write_started_us, frame_ready_interval_us,
        frame_acquire_duration_us, jpeg_metadata_prepare_duration_us,
        previous_response_write_valid ? "true" : "false",
        previous_frame_sequence, previous_response_write_duration_us,
        tof_timestamp_us, tof_minus_capture_us, tof_age_at_jpeg_ready_us,
        tof_during_acquire ? "true" : "false",
        tof_updates_during_acquire, tof_updates_since_previous_frame,
        kEnableTofSampling ? "true" : "false",
        range.valid ? "true" : "false", range.range_mm, range.status,
        range.range_status_code, static_cast<unsigned>(frame_length),
        static_cast<unsigned>(frame_width), static_cast<unsigned>(frame_height),
        camera_settings.jpeg_quality,
        camera_settings.auto_exposure ? "true" : "false",
        camera_psram_dma_enabled ? "true" : "false",
        stream_tcp_nodelay_enabled ? "true" : "false",
        kCoalesceStreamPreamble ? "true" : "false",
        kReuseStreamFrameCopyBuffer ? "true" : "false",
        static_cast<int>(xPortGetCoreID()),
        static_cast<unsigned>(uxTaskPriorityGet(nullptr)), exposure_value,
        wifi_rssi_dbm, free_heap_bytes);
    if (header_length <= 0 ||
        static_cast<size_t>(header_length) >= sizeof(header)) {
      if (!kReuseStreamFrameCopyBuffer) {
        heap_caps_free(frame_copy);
      }
      emitEvent("camera_stream", "FRAME_HEADER_OVERFLOW");
      stream_result = ESP_ERR_INVALID_SIZE;
      break;
    }
    esp_err_t result = ESP_OK;
    if (!kCoalesceStreamPreamble) {
      result = httpd_resp_send_chunk(request, kStreamBoundary,
                                     strlen(kStreamBoundary));
    }
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(
          request, header, static_cast<size_t>(header_length));
    }
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(
          request, reinterpret_cast<const char*>(frame_copy), frame_length);
    }
    const uint64_t response_write_completed_us = monotonicUs();
    if (!kReuseStreamFrameCopyBuffer) {
      heap_caps_free(frame_copy);
    }
    if (result != ESP_OK) {
      stream_result = result;
      break;
    }
    previous_jpeg_ready_us = jpeg_ready_timestamp_us;
    previous_range_update_count = range_updates_at_jpeg_ready;
    previous_frame_sequence = frame_sequence;
    previous_response_write_duration_us =
        response_write_completed_us - response_write_started_us;
    previous_response_write_valid = true;
    updateFrameStats(false, false, true);
  }
  heap_caps_free(reusable_frame_copy);
  updateFrameStats(true, false, false);
  return stream_result;
}

void registerUri(httpd_handle_t server, const char* uri,
                 httpd_method_t method, esp_err_t (*handler)(httpd_req_t*)) {
  httpd_uri_t route = {};
  route.uri = uri;
  route.method = method;
  route.handler = handler;
  httpd_register_uri_handler(server, &route);
}

bool startControlServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.max_uri_handlers = 12;
  if (httpd_start(&control_httpd, &config) != ESP_OK) {
    emitEvent("http_control", "START_FAILED");
    return false;
  }
  registerUri(control_httpd, "/", HTTP_GET, rootHandler);
  registerUri(control_httpd, "/status", HTTP_GET, statusPageHandler);
  registerUri(control_httpd, "/api/range", HTTP_GET, rangeHandler);
  registerUri(control_httpd, "/api/status", HTTP_GET, statusHandler);
  registerUri(control_httpd, "/api/time", HTTP_GET, timeSyncHandler);
  registerUri(control_httpd, "/api/camera", HTTP_GET,
              cameraSettingsGetHandler);
  registerUri(control_httpd, "/api/camera", HTTP_POST,
              cameraSettingsPostHandler);
  registerUri(control_httpd, "/api/snapshot", HTTP_GET, snapshotHandler);
  registerUri(control_httpd, "/save", HTTP_POST, saveWifiHandler);
  registerUri(control_httpd, "/forget", HTTP_POST, forgetWifiHandler);
  emitEvent("http_control", "READY_PORT_80");
  return true;
}

bool startStreamServer() {
  if (!camera_ready) {
    emitEvent("http_stream", "CAMERA_NOT_READY");
    return false;
  }
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;
  config.ctrl_port = 32769;
  config.max_open_sockets = 3;
  config.stack_size = 8192;
  config.core_id = kStreamServerCoreId;
  config.task_priority = kStreamServerTaskPriority;
  if (httpd_start(&stream_httpd, &config) != ESP_OK) {
    emitEvent("http_stream", "START_FAILED");
    return false;
  }
  registerUri(stream_httpd, "/stream", HTTP_GET, streamHandler);
  emitEvent("http_stream", "READY_PORT_81");
  return true;
}

bool loadWifiCredentials(String* ssid, String* password) {
  Preferences preferences;
  if (!preferences.begin(kPreferencesNamespace, true)) {
    return false;
  }
  *ssid = preferences.getString("ssid", "");
  *password = preferences.getString("password", "");
  preferences.end();
  return !ssid->isEmpty();
}

bool connectStoredWifi() {
  String ssid;
  String password;
  if (!loadWifiCredentials(&ssid, &password)) {
    emitEvent("wifi_station", "NO_STORED_CREDENTIALS");
    return false;
  }
  WiFi.mode(WIFI_STA);
  WiFi.persistent(false);
  WiFi.setAutoReconnect(true);
  WiFi.setSleep(false);
  WiFi.setHostname(kMdnsHostname);
  WiFi.begin(ssid.c_str(), password.c_str());
  const uint32_t started_ms = millis();
  while (WiFi.status() != WL_CONNECTED &&
         millis() - started_ms < kWifiConnectTimeoutMs) {
    delay(250);
  }
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.disconnect(true, false);
    emitEvent("wifi_station", "CONNECT_TIMEOUT");
    return false;
  }
  emitEvent("wifi_station", "CONNECTED");
  wifi_was_connected = true;
  return true;
}

void startSetupAccessPoint() {
  setup_mode = true;
  WiFi.mode(WIFI_AP);
  if (WiFi.softAP(kSetupSsid, kSetupPassword, 6, false, 2)) {
    emitEvent("wifi_setup_ap", "READY_192_168_4_1");
  } else {
    emitEvent("wifi_setup_ap", "START_FAILED");
  }
  startTimingUdp();
  startControlServer();
}

void startNetworkServices() {
  if (!connectStoredWifi()) {
    startSetupAccessPoint();
    return;
  }
  setup_mode = false;
  if (MDNS.begin(kMdnsHostname)) {
    MDNS.addService("http", "tcp", 80);
    emitEvent("mdns", "READY_ATOMS3R_TOF_LOCAL");
  } else {
    emitEvent("mdns", "START_FAILED");
  }
  startControlServer();
  startStreamServer();
  startTimingUdp();
  emitEvent("dashboard", WiFi.localIP().toString().c_str());
  next_wifi_reconnect_us =
      static_cast<uint64_t>(esp_timer_get_time()) +
      static_cast<uint64_t>(kWifiReconnectIntervalMs) * 1000ULL;
}

void maintainWifi() {
  if (setup_mode) {
    return;
  }
  const bool connected = WiFi.status() == WL_CONNECTED;
  if (connected) {
    if (!wifi_was_connected) {
      wifi_was_connected = true;
      emitEvent("wifi_station", "RECONNECTED");
    }
    return;
  }
  if (wifi_was_connected) {
    wifi_was_connected = false;
    emitEvent("wifi_station", "DISCONNECTED_AUTO_RETRY");
  }
  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  if (now_us < next_wifi_reconnect_us) {
    return;
  }
  ++wifi_reconnect_attempts;
  WiFi.reconnect();
  next_wifi_reconnect_us =
      now_us + static_cast<uint64_t>(kWifiReconnectIntervalMs) * 1000ULL;
}

void emitRuntimeHealth() {
  if (setup_mode) {
    if (camera_ready && sensor_ready) {
      emitEvent("runtime_health", "SETUP_AP_CAMERA_READY_TOF_READY");
    } else if (!camera_ready && sensor_ready) {
      emitEvent("camera_error", camera_failure_status);
      emitEvent("runtime_health", "SETUP_AP_CAMERA_FAILED_TOF_READY");
    } else {
      emitEvent("runtime_health", "SETUP_AP_NOT_FULLY_READY");
    }
    return;
  }
  if (WiFi.status() == WL_CONNECTED && camera_ready && sensor_ready) {
    emitEvent("runtime_health", "STATION_CAMERA_READY_TOF_READY");
  } else {
    if (!camera_ready) {
      emitEvent("camera_error", camera_failure_status);
    }
    emitEvent("runtime_health", "STATION_NOT_FULLY_READY");
  }
}

}  // namespace

void setup() {
  Serial.begin(kSerialBaud);
  pinMode(kCameraPowerPin, OUTPUT);
  digitalWrite(kCameraPowerPin, LOW);
  delay(1500);

  const uint64_t chip_id = ESP.getEfuseMac();
  const uint32_t boot_nonce = esp_random();
  snprintf(sequence_id, sizeof(sequence_id), "%012" PRIx64 "-%08" PRIx32,
           chip_id, boot_nonce);
  snprintf(clock_domain, sizeof(clock_domain), "esp32_boot_monotonic:%s",
           sequence_id);

  tof_bus.begin(kSdaPin, kSclPin, kI2cFrequencyHz);
  emitEvent("boot", "READY_FOR_TOF_INIT");
  sensor_ready = initializeSensor();
  next_retry_us = static_cast<uint64_t>(esp_timer_get_time()) + 1000000ULL;
  camera_mutex = xSemaphoreCreateMutex();
  if (camera_mutex == nullptr) {
    snprintf(camera_failure_status, sizeof(camera_failure_status),
             "CAMERA_MUTEX_ALLOCATION_FAILED");
    emitEvent("camera_init", camera_failure_status);
  } else {
    camera_ready = initializeCamera();
  }
  startNetworkServices();
  next_health_event_us =
      static_cast<uint64_t>(esp_timer_get_time()) + 5000000ULL;
}

void loop() {
  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  maintainWifi();
  if (now_us >= next_health_event_us) {
    emitRuntimeHealth();
    next_health_event_us = now_us + 5000000ULL;
  }
  if (sensor_ready) {
    if (kEnableTofSampling) {
      emitSample();
    } else {
      delay(10);
    }
    return;
  }

  if (now_us >= next_retry_us) {
    sensor_ready = initializeSensor();
    next_retry_us = now_us + 1000000ULL;
  }
  delay(10);
}
