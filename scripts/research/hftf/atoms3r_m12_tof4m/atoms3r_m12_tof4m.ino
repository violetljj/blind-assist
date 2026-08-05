#include <Arduino.h>
#include <ESPmDNS.h>
#include <Preferences.h>
#include <VL53L1X.h>
#include <WiFi.h>
#include <Wire.h>
#include <esp_camera.h>
#include <esp_http_server.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <inttypes.h>

#include "web_ui.h"

namespace {

constexpr char kFirmwareVersion[] = "atoms3r_m12_tof4m_web_r1";
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
char camera_failure_status[48] = "NOT_ATTEMPTED";

struct SharedRange {
  bool valid;
  uint16_t range_mm;
  uint8_t range_status_code;
  uint64_t timestamp_ns;
  const char* status;
};

SharedRange shared_range = {false, 0, 0, 0, "NOT_READY"};
portMUX_TYPE range_mux = portMUX_INITIALIZER_UNLOCKED;

constexpr char kStreamContentType[] =
    "multipart/x-mixed-replace;boundary=123456789000000000000987654321";
constexpr char kStreamBoundary[] =
    "\r\n--123456789000000000000987654321\r\n";
constexpr char kStreamPart[] =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

uint64_t monotonicNs() {
  return static_cast<uint64_t>(esp_timer_get_time()) * 1000ULL;
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
  config.frame_size = FRAMESIZE_XGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;
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

  sensor_t* camera_sensor = esp_camera_sensor_get();
  if (camera_sensor != nullptr && camera_sensor->id.PID == OV3660_PID) {
    camera_sensor->set_vflip(camera_sensor, 1);
    camera_sensor->set_brightness(camera_sensor, 1);
    camera_sensor->set_saturation(camera_sensor, -2);
  }
  emitEvent("camera_init", "READY_XGA_JPEG_Q10");
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

esp_err_t rangeHandler(httpd_req_t* request) {
  SharedRange snapshot;
  portENTER_CRITICAL(&range_mux);
  snapshot = shared_range;
  portEXIT_CRITICAL(&range_mux);

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
  httpd_resp_set_type(request, "application/json");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(request, body, HTTPD_RESP_USE_STRLEN);
}

esp_err_t saveWifiHandler(httpd_req_t* request) {
  if (request->content_len <= 0 || request->content_len > 256) {
    return httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST,
                               "invalid form length");
  }
  char body[257] = {};
  int received = 0;
  while (received < request->content_len) {
    const int result = httpd_req_recv(request, body + received,
                                      request->content_len - received);
    if (result <= 0) {
      return ESP_FAIL;
    }
    received += result;
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
  while (true) {
    camera_fb_t* frame = esp_camera_fb_get();
    if (frame == nullptr) {
      emitEvent("camera_capture", "FRAME_FAILED");
      return ESP_FAIL;
    }
    char header[96];
    const size_t header_length =
        snprintf(header, sizeof(header), kStreamPart,
                 static_cast<unsigned>(frame->len));
    esp_err_t result =
        httpd_resp_send_chunk(request, kStreamBoundary, strlen(kStreamBoundary));
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(request, header, header_length);
    }
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(
          request, reinterpret_cast<const char*>(frame->buf), frame->len);
    }
    esp_camera_fb_return(frame);
    if (result != ESP_OK) {
      return result;
    }
  }
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
  config.max_uri_handlers = 8;
  if (httpd_start(&control_httpd, &config) != ESP_OK) {
    emitEvent("http_control", "START_FAILED");
    return false;
  }
  registerUri(control_httpd, "/", HTTP_GET, rootHandler);
  registerUri(control_httpd, "/api/range", HTTP_GET, rangeHandler);
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
  emitEvent("dashboard", WiFi.localIP().toString().c_str());
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
  camera_ready = initializeCamera();
  startNetworkServices();
  next_health_event_us =
      static_cast<uint64_t>(esp_timer_get_time()) + 5000000ULL;
}

void loop() {
  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  if (now_us >= next_health_event_us) {
    emitRuntimeHealth();
    next_health_event_us = now_us + 5000000ULL;
  }
  if (sensor_ready) {
    emitSample();
    return;
  }

  if (now_us >= next_retry_us) {
    sensor_ready = initializeSensor();
    next_retry_us = now_us + 1000000ULL;
  }
  delay(10);
}
