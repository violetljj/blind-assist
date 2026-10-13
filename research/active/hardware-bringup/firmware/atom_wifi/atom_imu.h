#pragma once
#include <Wire.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "atom_bmi270_config.h"
#include "atom_imu_filter.h"
#include "atom_imu_history.h"

// AtomS3R-M12 internal I2C: SDA45/SCL0 (M5Stack schematic and M5Unified).
// With explicit SCCB pins, esp_camera ignores camera_config.sccb_i2c_port and
// uses CONFIG_SCCB_HARDWARE_I2C_PORT1 (true in the pinned ESP32S3 SDK). Reserve
// that actual camera bus and put BMI270 on the other I2C controller.
// Registers follow Bosch BST-BMI270-DS000-05 Rev1.3, sections4.3 and5.
namespace AtomImu {
static constexpr uint16_t port = 3336;
static constexpr const char* version = "atom-bmi270-v1";
#if CONFIG_SCCB_HARDWARE_I2C_PORT1
static constexpr uint8_t cameraSccbPort=1;
static constexpr uint8_t sensorPort=0;
#else
static constexpr uint8_t cameraSccbPort=0;
static constexpr uint8_t sensorPort=1;
#endif
static_assert(cameraSccbPort!=sensorPort,"Camera and BMI270 must own separate I2C buses");
static TwoWire sensorWire(sensorPort);
static uint8_t address = 0x68;
static bool ready = false;
static const char* initStatus = "not_initialized";
static uint32_t readErrors = 0;
static portMUX_TYPE sampleMux = portMUX_INITIALIZER_UNLOCKED;
struct Sample {
  uint32_t seq = 0, dtUs = 0, readUs = 0;
  int64_t deviceUs = 0;
  int16_t accelRaw[3] = {}, gyroRaw[3] = {};
  float accelG[3] = {}, gyroDps[3] = {}, gravity[3] = {0,0,1};
  float pitch = 0, roll = 0;
  bool valid = false, accelCorrection = false, gap = false;
};
static Sample latest;
static constexpr size_t historySamples=64;
static constexpr int64_t historyMaxAgeUs=500000;
using SampleHistory=AtomImuHistory<Sample,historySamples>;
static SampleHistory history;
// Transport is serviced only by the Arduino loop, as is the HTTP status handler.
// These cumulative counters include consumers that have since disconnected.
static uint64_t historyOverwritten=0,historyExpired=0;

static bool writeRegister(uint8_t reg, const uint8_t* bytes, size_t count) {
  sensorWire.beginTransmission(address);
  sensorWire.write(reg);
  bool complete = sensorWire.write(bytes,count)==count;
  return sensorWire.endTransmission()==0 && complete;
}
static bool writeByte(uint8_t reg, uint8_t value) { return writeRegister(reg,&value,1); }
static bool readRegister(uint8_t reg, uint8_t* bytes, size_t count) {
  sensorWire.beginTransmission(address);sensorWire.write(reg);
  if (sensorWire.endTransmission(false)!=0) return false;
  if (sensorWire.requestFrom(address,count,true)!=count) {
    while(sensorWire.available()) sensorWire.read();
    return false;
  }
  for(size_t i=0;i<count;++i) bytes[i]=sensorWire.read();
  return true;
}
static bool initialize() {
  if (!sensorWire.begin(45,0,400000)) { initStatus="i2c_begin_failed";return false; }
  sensorWire.setTimeOut(20);
  uint8_t chip=0;
  if (!readRegister(0x00,&chip,1) || chip!=0x24) {
    address=0x69;
    if (!readRegister(0x00,&chip,1) || chip!=0x24) { initStatus="bmi270_not_found";return false; }
  }
  if (!writeByte(0x7E,0xB6)) { initStatus="soft_reset_failed";return false; }
  delay(10);
  if (!writeByte(0x7C,0x00)) { initStatus="power_config_failed";return false; }
  delay(1); // >=450us after disabling advanced power save.
  if (!writeByte(0x59,0x00)) { initStatus="config_prepare_failed";return false; }
  for(size_t i=0;i<sizeof(atomBmi270Config);i+=32) {
    uint8_t offset[2]={(uint8_t)((i>>1)&0x0F),(uint8_t)(i>>5)};
    if (!writeRegister(0x5B,offset,2) || !writeRegister(0x5E,atomBmi270Config+i,32)) {
      initStatus="config_upload_failed";return false;
    }
  }
  if (!writeByte(0x59,0x01)) { initStatus="config_complete_failed";return false; }
  delay(25);
  uint8_t internal=0;
  if (!readRegister(0x21,&internal,1) || (internal&0x0F)!=1) {
    initStatus="bmi270_internal_status_error";return false;
  }
  // Both ODR100Hz, performance filter + normal bandwidth; native +/-4g and
  // +/-500deg/s. No magnetometer, hardware offset compensation or axis remap.
  if (!writeByte(0x41,0x01) || !writeByte(0x43,0x02) ||
      !writeByte(0x40,0xA8) || !writeByte(0x42,0xA8) || !writeByte(0x7D,0x06)) {
    initStatus="sensor_config_failed";return false;
  }
  delay(80);
  uint8_t config[4]={};
  if (!readRegister(0x40,config,4) || config[0]!=0xA8 || config[1]!=0x01 ||
      config[2]!=0xA8 || config[3]!=0x02) { initStatus="config_readback_failed";return false; }
  initStatus="ok";return true;
}
static Sample snapshot(uint32_t* errors=nullptr) {
  portENTER_CRITICAL(&sampleMux);
  Sample copy=latest;if(errors)*errors=readErrors;
  portEXIT_CRITICAL(&sampleMux);
  return copy;
}
static void sensorTask(void*) {
  AtomGravityFilter filter;
  uint32_t seq=0;int64_t previous=0;
  TickType_t wake=xTaskGetTickCount();
  const TickType_t pollTicks=pdMS_TO_TICKS(5)>0?pdMS_TO_TICKS(5):1;
  while(true) {
    // Poll at200Hz so reads straddling the sensor100Hz update are not treated as
    // duplicate samples. Only consume when BOTH fresh-data bits are present.
    vTaskDelayUntil(&wake,pollTicks);
    uint8_t status=0,bytes[12];int64_t started=esp_timer_get_time();
    if (!readRegister(0x03,&status,1)) {
      portENTER_CRITICAL(&sampleMux);++readErrors;portEXIT_CRITICAL(&sampleMux);continue;
    }
    if ((status&0xC0)!=0xC0) continue;
    if (!readRegister(0x0C,bytes,sizeof(bytes))) {
      portENTER_CRITICAL(&sampleMux);++readErrors;portEXIT_CRITICAL(&sampleMux);continue;
    }
    Sample s;s.seq=++seq;s.deviceUs=esp_timer_get_time();
    s.readUs=(uint32_t)(s.deviceUs-started);
    s.dtUs=previous?(uint32_t)(s.deviceUs-previous):0;
    s.gap=previous && s.dtUs>30000;previous=s.deviceUs;
    for(int i=0;i<3;++i) {
      s.accelRaw[i]=(int16_t)((uint16_t)bytes[i*2]|((uint16_t)bytes[i*2+1]<<8));
      s.gyroRaw[i]=(int16_t)((uint16_t)bytes[6+i*2]|((uint16_t)bytes[7+i*2]<<8));
      s.accelG[i]=s.accelRaw[i]*(4.0f/32768.0f);
      s.gyroDps[i]=s.gyroRaw[i]*(500.0f/32768.0f);
    }
    s.valid=filter.update(s.accelG,s.gyroDps,s.dtUs*1e-6f,s.gap);
    s.accelCorrection=filter.accelCorrection;
    for(int i=0;i<3;++i)s.gravity[i]=filter.gravity[i];
    s.pitch=filter.pitchDeg();s.roll=filter.rollDeg();
    portENTER_CRITICAL(&sampleMux);latest=s;history.publish(s);portEXIT_CRITICAL(&sampleMux);
  }
}
static int bootJson(char* out,size_t capacity) {
  return snprintf(out,capacity,
    "{\"type\":\"imu_boot\",\"schema_version\":1,\"firmware\":\"%s\",\"imu_version\":\"%s\",\"device_id\":\"%s\",\"boot_id\":\"%s\","
    "\"sensor\":\"BMI270\",\"sample_hz\":100,\"accel_range_g\":4,\"gyro_range_dps\":500,\"accel_lsb_per_g\":8192,\"gyro_scale_dps_per_lsb\":0.0152587890625,"
    "\"filter\":\"gravity_complementary_v1\",\"filter_tau_s\":0.5,\"axis_convention\":\"bmi270_native_right_handed\","
    "\"pitch_formula\":\"atan2(-ax,sqrt(ay^2+az^2))\",\"roll_formula\":\"atan2(ay,az)\",\"mount_calibration\":\"host_side_required\","
    "\"clock_domain\":\"ATOM_MONOTONIC_UNMAPPED\",\"timestamp_semantics\":\"atom_register_read_complete\",\"imu_port\":3336,\"ready\":%s,\"init_status\":\"%s\"}\n",
    demoFirmware,version,deviceId.c_str(),bootId.c_str(),ready?"true":"false",initStatus);
}
static int frameJson(const Sample& s,char* out,size_t capacity) {
  return snprintf(out,capacity,
    "{\"type\":\"imu_frame\",\"schema_version\":1,\"device_id\":\"%s\",\"boot_id\":\"%s\",\"seq\":%lu,\"device_us\":%lld,\"dt_us\":%lu,\"read_us\":%lu,"
    "\"accel_raw\":[%d,%d,%d],\"gyro_raw\":[%d,%d,%d],\"accel_g\":[%.6f,%.6f,%.6f],\"gyro_dps\":[%.6f,%.6f,%.6f],"
    "\"gravity_unit\":[%.7f,%.7f,%.7f],\"pitch_deg\":%.5f,\"roll_deg\":%.5f,\"valid\":%s,\"accel_correction\":%s,\"sample_gap\":%s}\n",
    deviceId.c_str(),bootId.c_str(),(unsigned long)s.seq,(long long)s.deviceUs,(unsigned long)s.dtUs,(unsigned long)s.readUs,
    s.accelRaw[0],s.accelRaw[1],s.accelRaw[2],s.gyroRaw[0],s.gyroRaw[1],s.gyroRaw[2],
    s.accelG[0],s.accelG[1],s.accelG[2],s.gyroDps[0],s.gyroDps[1],s.gyroDps[2],
    s.gravity[0],s.gravity[1],s.gravity[2],s.pitch,s.roll,s.valid?"true":"false",s.accelCorrection?"true":"false",s.gap?"true":"false");
}
static int statusJson(char* out,size_t capacity) {
  uint32_t errors;Sample s=snapshot(&errors);
  return snprintf(out,capacity,
    "{\"type\":\"imu_status\",\"schema_version\":1,\"device_id\":\"%s\",\"boot_id\":\"%s\",\"device_us\":%lld,\"ready\":%s,\"init_status\":\"%s\",\"read_errors\":%lu,\"last_seq\":%lu,\"sample_age_us\":%lld,"
    "\"transport_history_samples\":64,\"transport_max_age_us\":500000,\"transport_overwritten_samples\":%llu,\"transport_expired_samples\":%llu}\n",
    deviceId.c_str(),bootId.c_str(),(long long)esp_timer_get_time(),ready?"true":"false",initStatus,(unsigned long)errors,
    (unsigned long)s.seq,s.deviceUs?(long long)(esp_timer_get_time()-s.deviceUs):-1LL,
    (unsigned long long)historyOverwritten,(unsigned long long)historyExpired);
}
static void begin() {
  ready=initialize();
  if (ready && xTaskCreatePinnedToCore(sensorTask,"bmi270_sampler",4096,nullptr,3,nullptr,1)!=pdPASS) {
    ready=false;initStatus="sensor_task_start_failed";
  }
  char config[1200];int n=bootJson(config,sizeof(config));
  if(n>0 && n<(int)sizeof(config)) Serial.print(config);
  api.on("/api/imu",HTTP_GET,[](){
    // The HTTP handler runs on the existing Arduino loop; retain its small
    // stack for camera/network work. This handler is invoked sequentially.
    static char output[3072],config[1200],frame[1000],status[600];
    Sample s=snapshot();bootJson(config,sizeof(config));frameJson(s,frame,sizeof(frame));statusJson(status,sizeof(status));
    // Objects contain trailing newlines, which JSON permits as whitespace.
    int count=snprintf(output,sizeof(output),"{\"config\":%s,\"status\":%s,\"latest\":%s}",config,status,s.seq?frame:"null");
    if(count<=0 || count>=(int)sizeof(output)){api.send(500,"application/json","{\"error\":\"imu_status_format_failed\"}");return;}
    api.sendHeader("Cache-Control","no-store");api.send(200,"application/json",output);
  });
}

// Nonblocking2-client transport: shared64-sample history and one immutable
// pending packet per consumer. Brief send stalls retain intermediate samples;
// stale/overwritten samples remain visible as raw seq gaps and status counters.
// Sampling never waits for WiFi, a browser, a TCP client, or Serial output.
struct Consumer {
  int socket=-1;char pending[1200]={};size_t length=0,offset=0;
  uint32_t statusMs=0,progressMs=0;
  SampleHistory::Cursor cursor;
};
static Consumer clients[2];static int listener=-1;
// Called only by the Arduino loop, which also owns these client sockets.
static bool hasTransportClients() {
  for(const auto& c:clients)if(c.socket>=0)return true;
  return false;
}
static void closeConsumer(Consumer& c) { if(c.socket>=0)close(c.socket);c=Consumer(); }
static void closeTransport() {
  for(auto& c:clients)closeConsumer(c);
  if(listener>=0)close(listener);listener=-1;
}
static bool queued(Consumer& c,int count) {
  if(count<=0 || count>=(int)sizeof(c.pending)){closeConsumer(c);return false;}
  c.length=count;c.offset=0;c.progressMs=millis();return true;
}
static void loop() {
  if(WiFi.status()!=WL_CONNECTED){closeTransport();return;}
  if(listener<0) {
    listener=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);if(listener<0)return;
    int one=1;setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
    fcntl(listener,F_SETFL,O_NONBLOCK);
    sockaddr_in addr={};addr.sin_family=AF_INET;addr.sin_port=htons(port);addr.sin_addr.s_addr=htonl(INADDR_ANY);
    if(bind(listener,(sockaddr*)&addr,sizeof(addr))!=0 || listen(listener,2)!=0){closeTransport();return;}
  }
  int incoming=accept(listener,nullptr,nullptr);
  if(incoming>=0) {
    Consumer* slot=nullptr;for(auto& c:clients)if(c.socket<0){slot=&c;break;}
    if(!slot)close(incoming);
    else {
      slot->socket=incoming;fcntl(incoming,F_SETFL,O_NONBLOCK);
      int one=1;setsockopt(incoming,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
      portENTER_CRITICAL(&sampleMux);history.startLatest(slot->cursor);portEXIT_CRITICAL(&sampleMux);
      queued(*slot,bootJson(slot->pending,sizeof(slot->pending)));
    }
  }
  Sample s=snapshot();int64_t nowUs=esp_timer_get_time();
  for(auto& c:clients) {
    if(c.socket<0)continue;
    // Drain at most one small client write; this is an output-only protocol.
    char discard[32];int received=recv(c.socket,discard,sizeof(discard),MSG_DONTWAIT);
    if(received==0 || (received<0 && errno!=EAGAIN && errno!=EWOULDBLOCK)){closeConsumer(c);continue;}
    if(!c.length) {
      Sample next;
      const uint64_t overwritten=c.cursor.overwritten,expired=c.cursor.expired;
      portENTER_CRITICAL(&sampleMux);
      bool available=history.take(c.cursor,nowUs,historyMaxAgeUs,next);
      portEXIT_CRITICAL(&sampleMux);
      historyOverwritten+=c.cursor.overwritten-overwritten;
      historyExpired+=c.cursor.expired-expired;
      if(available) {
        queued(c,frameJson(next,c.pending,sizeof(c.pending)));
      } else if((!ready || !s.seq || nowUs-s.deviceUs>50000) && (uint32_t)(millis()-c.statusMs)>=1000) {
        c.statusMs=millis();queued(c,statusJson(c.pending,sizeof(c.pending)));
      }
    }
    if(c.socket<0 || !c.length)continue;
    int sent=send(c.socket,c.pending+c.offset,c.length-c.offset,MSG_DONTWAIT);
    if(sent>0) {
      c.offset+=sent;c.progressMs=millis();
      if(c.offset==c.length)c.length=c.offset=0;
    } else if((sent<0 && errno!=EAGAIN && errno!=EWOULDBLOCK) || (uint32_t)(millis()-c.progressMs)>500) {
      closeConsumer(c);
    }
  }
}
} // namespace AtomImu
