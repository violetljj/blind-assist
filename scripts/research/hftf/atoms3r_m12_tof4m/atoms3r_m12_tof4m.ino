#include <Arduino.h>
#include <Wire.h>
#include <VL53L1X.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <inttypes.h>

namespace {

constexpr char kFirmwareVersion[] = "atoms3r_m12_tof4m_r0";
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

VL53L1X sensor;
bool sensor_ready = false;
uint64_t sample_index = 0;
uint64_t next_retry_us = 0;
char sequence_id[40] = {};
char clock_domain[64] = {};

uint64_t monotonicNs() {
  return static_cast<uint64_t>(esp_timer_get_time()) * 1000ULL;
}

bool probeTofAddress() {
  Wire.beginTransmission(kTofAddress);
  return Wire.endTransmission() == 0;
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

bool initializeSensor() {
  if (!probeTofAddress()) {
    emitEvent("i2c_probe", "TOF4M_0X29_NOT_FOUND");
    return false;
  }
  emitEvent("i2c_probe", "TOF4M_0X29_FOUND");

  sensor.setBus(&Wire);
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

}  // namespace

void setup() {
  Serial.begin(kSerialBaud);
  delay(1000);

  const uint64_t chip_id = ESP.getEfuseMac();
  const uint32_t boot_nonce = esp_random();
  snprintf(sequence_id, sizeof(sequence_id), "%012" PRIx64 "-%08" PRIx32,
           chip_id, boot_nonce);
  snprintf(clock_domain, sizeof(clock_domain), "esp32_boot_monotonic:%s",
           sequence_id);

  Wire.begin(kSdaPin, kSclPin, kI2cFrequencyHz);
  emitEvent("boot", "READY_FOR_TOF_INIT");
  sensor_ready = initializeSensor();
  next_retry_us = static_cast<uint64_t>(esp_timer_get_time()) + 1000000ULL;
}

void loop() {
  if (sensor_ready) {
    emitSample();
    return;
  }

  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  if (now_us >= next_retry_us) {
    sensor_ready = initializeSensor();
    next_retry_us = now_us + 1000000ULL;
  }
  delay(10);
}
