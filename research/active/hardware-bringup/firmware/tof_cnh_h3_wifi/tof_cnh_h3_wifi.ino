// Transport-only H3 variant: local WiFi TCP3335 + optional USB, no model changes.
// 8x8-zone CNH in the simulator's H3 layout: start_bin 0, 16 bins, sub_sample 8
// (0.30 m bins to 4.8 m), 5 Hz autonomous, 20 ms per-zone integration as tof_cnh.
// Bring-up/frame-rate check only; same JSON framing as tof_cnh with 8x8 arrays.
#include <Arduino.h>
#include <Wire.h>
extern "C" {
#include "vl53lmz_api.h"
#include "vl53lmz_plugin_cnh.h"
}

// Explicit C++ prototypes keep Arduino's sketch preprocessor from carrying the
// vendor C-linkage wrapper onto setup/loop or the transport helper functions.
static bool sendCnhClient(const String& text);
static void serviceCnhClient();
static bool ok(const char* op, uint8_t s);
void setup();
void loop();

#include <WiFi.h>
#include <WiFiUdp.h>
#include <WebServer.h>
#include <Preferences.h>
#include <lwip/sockets.h>
static WiFiServer cnhServer(3335);
static WiFiClient cnhClient;

// One complete NDJSON line at a time; the histogram/calibration is unchanged.
// No sample backlog. A stalled/partial TCP consumer is disconnected visibly.
// A phone received only 5744 bytes of its first CNH frame before EOF under the
// old 30 ms budget. Allow delayed ACK/backpressure for
// up to two 5 Hz periods, but stop after one period without any byte progress.
static constexpr uint32_t CNH_TCP_ROW_DEADLINE_US = 400000;
static constexpr uint32_t CNH_TCP_STALL_DEADLINE_US = 200000;
static bool sendCnhClient(const String& text) {
  if (!cnhClient.connected()) return false;
  const uint8_t* data = (const uint8_t*)text.c_str();
  size_t sent = 0;
  const uint32_t started = micros();
  uint32_t lastProgress = started;
  while (sent < text.length()) {
    const uint32_t now = micros();
    if ((uint32_t)(now-started) >= CNH_TCP_ROW_DEADLINE_US ||
        (uint32_t)(now-lastProgress) >= CNH_TCP_STALL_DEADLINE_US) break;
    int count = send(cnhClient.fd(), data+sent, text.length()-sent, MSG_DONTWAIT);
    if (count > 0) {
      sent += count;
      lastProgress = micros();
    }
    else if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) break;
    else delay(1);
  }
  if (sent != text.length()) { cnhClient.stop(); return false; }
  return true;
}

class CnhMirror : public Print {
  String line;
  bool overflow = false;
 public:
  String bootRecord, configRecord;
  void begin(unsigned long baud) {
    Serial.begin(baud);
    Serial.setTxTimeoutMs(5);
    line.reserve(32768);
    bootRecord.reserve(512);
    configRecord.reserve(512);
  }
  int available() { return Serial.available(); }
  int read() { return Serial.read(); }
  size_t write(uint8_t value) override {
    if (!overflow) {
      if (line.length() >= 32768) overflow = true;
      else line += (char)value;
    }
    if (value == '\n') {
      if (overflow) {
        line = "{\"type\":\"error\",\"message\":\"cnh_line_overflow\"}\n";
      }
      if (line.startsWith("{\"type\":\"boot\"")) bootRecord = line;
      if (line.startsWith("{\"type\":\"config\"")) configRecord = line;
      // Native USB CDC is optional: no waiting for a host on powerbank power.
      if (Serial) Serial.write((const uint8_t*)line.c_str(), line.length());
      sendCnhClient(line);
      line.remove(0);
      overflow = false;
    }
    return 1;
  }
  size_t write(const uint8_t* data, size_t count) override {
    for (size_t i=0; i<count; ++i) write(data[i]);
    return count;
  }
};
static CnhMirror cnhOutput;
// Use existing provisioning/discovery/NVS unchanged; redirect its output only.
#define Serial cnhOutput
#include "demo_wifi.h"
#undef Serial

static void serviceCnhClient() {
  if (WiFi.status()!=WL_CONNECTED) { cnhClient.stop(); return; }
  WiFiClient candidate = cnhServer.accept();
  if (!candidate) return;
  if (cnhClient.connected()) { candidate.stop(); return; }
  cnhClient.stop();
  cnhClient = candidate;
  cnhClient.setNoDelay(true);
  String hello = String("{\"type\":\"transport_connection\",\"transport\":\"tcp\",\"port\":3335,")+
      "\"is_physical_boot\":false,\"startup_kind\":\"transport_only_connection\",\"boot_id\":\""+
      bootId+"\",\"device_id\":\""+deviceId+"\",\"cached_boot_config_follow\":true}\n";
  if (!sendCnhClient(hello)) return;
  // Replay the actual startup records with the SAME physical boot_id. A new
  // TCP connection must never be interpreted as a hardware restart.
  if (!cnhOutput.bootRecord.isEmpty()) sendCnhClient(cnhOutput.bootRecord);
  if (!cnhOutput.configRecord.isEmpty()) sendCnhClient(cnhOutput.configRecord);
}

#ifndef CNH_I2C_HZ
#define CNH_I2C_HZ 1000000  // 400 kHz measured 3.81 Hz (169.5 ms per read) on 2026-09-27
#endif
static const int Z=64, NB=16;
static VL53LMZ_Configuration dev;
static VL53LMZ_ResultsData result;
static VL53LMZ_Motion_Configuration cfg;
static cnh_data_buffer_t cnh;
static uint32_t bytes=0, seq=0, lastFrame=0, lastNote=0, readUs=0;
static bool running=false;
static const char* lastOp="none";
static unsigned lastStatus=0;
static bool ok(const char* op, uint8_t s) {
  cnhOutput.printf("{\"type\":\"step\",\"op\":\"%s\",\"status\":%u}\n",op,s);
  lastOp=op; lastStatus=s;
  return s==0;
}
void setup() {
  cnhOutput.begin(115200); delay(2000);
  beginDemoNetwork("tof","xiao-cnh-h3-wifi-v1");
  cnhServer.begin();
  api.on("/api/cnh",HTTP_GET,[]{
    String identity=demoIdentity(); identity.remove(identity.length()-1);
    String payload=identity+",\"transport\":\"tcp_ndjson\",\"tcp_port\":3335,\"firmware_config\":";
    payload+=cnhOutput.configRecord.isEmpty()?"null":cnhOutput.configRecord;
    payload+="}";
    api.sendHeader("Cache-Control","no-store");api.send(200,"application/json",payload);
  });
  cnhOutput.printf("{\"type\":\"boot\",\"firmware\":\"xiao-cnh-h3-wifi-v1\",\"startup_kind\":\"physical_boot\",\"boot_id\":\"%s\",\"device_id\":\"%s\"}\n",bootId.c_str(),deviceId.c_str());
  Wire.begin(5,6,CNH_I2C_HZ); Wire.setTimeOut(100);
  dev.platform.address=0x52;
  uint8_t alive=0;
  if(!ok("is_alive",vl53lmz_is_alive(&dev,&alive)) || !alive) return;
  if(!ok("init",vl53lmz_init(&dev))) return;
  if(!ok("resolution",vl53lmz_set_resolution(&dev,Z))) return;
  if(!ok("mode",vl53lmz_set_ranging_mode(&dev,VL53LMZ_RANGING_MODE_AUTONOMOUS))) return;
  if(!ok("frequency",vl53lmz_set_ranging_frequency_hz(&dev,5))) return;
  if(!ok("integration",vl53lmz_set_integration_time_ms(&dev,20))) return;
  if(!ok("cnh_init",vl53lmz_cnh_init_config(&cfg,0,NB,8))) return;
  if(!ok("cnh_map",vl53lmz_cnh_create_agg_map(&cfg,Z,0,0,1,1,8,8))) return;
  if(!ok("cnh_memory",vl53lmz_cnh_calc_required_memory(&cfg,&bytes))) return;
  if(bytes>sizeof(cnh) || bytes%4) {cnhOutput.printf("{\"type\":\"error\",\"message\":\"cnh_size\",\"bytes\":%lu}\n",bytes);return;}
  if(!ok("cnh_config",vl53lmz_cnh_send_config(&dev,&cfg))) return;
  if(!ok("output_config",vl53lmz_create_output_config(&dev))) return;
  union Block_header bh;
  bh.idx=VL53LMZ_CNH_DATA_IDX; bh.type=4; bh.size=bytes/4;
  if(!ok("cnh_block",vl53lmz_add_output_block(&dev,bh.bytes))) return;
  if(!ok("start",vl53lmz_send_output_config_and_start(&dev))) return;
  cnhOutput.printf("{\"type\":\"config\",\"zones\":%d,\"bins\":%d,\"start_bin\":0,\"sub_sample\":8,\"base_bin_mm\":37.5348,\"i2c_hz\":%lu,\"cnh_bytes\":%lu,\"read_bytes\":%lu,\"frequency_hz\":5,\"integration_ms\":20,\"ranging_mode\":\"autonomous\",\"transport\":\"tcp_ndjson\",\"tcp_port\":3335}\n",
                Z,NB,(unsigned long)CNH_I2C_HZ,bytes,dev.data_read_size);
  running=true; lastFrame=millis();
}
void loop() {
  demoNetworkLoop();
  serviceCnhClient();
  if(!running) {cnhOutput.printf("{\"type\":\"error\",\"message\":\"init_failed\",\"op\":\"%s\",\"status\":%u,\"cnh_bytes\":%lu}\n",lastOp,lastStatus,bytes);delay(3000);return;}
  uint8_t ready=0, s=vl53lmz_check_data_ready(&dev,&ready);
  if(s) {ok("ready",s);delay(500);return;}
  if(ready) {
    uint32_t t0=micros();
    s=vl53lmz_get_ranging_data(&dev,&result);
    readUs=micros()-t0;
    if(s) {ok("read",s);delay(100);return;}
    s=vl53lmz_results_extract_block(&dev,VL53LMZ_CNH_DATA_IDX,(uint8_t*)cnh,bytes);
    if(s) {ok("extract",s);delay(100);return;}
    // Validate header and pointer ranges before dereferencing variable CNH blocks.
    if((cnh[4]&0xffff)!=Z || (cnh[4]>>16)!=NB) {
      cnhOutput.printf("{\"type\":\"error\",\"message\":\"cnh_header\",\"value\":%lu}\n",cnh[4]);delay(500);return;
    }
    int32_t *hist[Z], *ambient[Z]; int8_t *scale[Z], *ambScale[Z];
    uintptr_t begin=(uintptr_t)cnh, end=begin+bytes;
    for(int a=0;a<Z;++a) {
      vl53lmz_cnh_get_block_addresses(&cfg,a,cnh,&hist[a],&scale[a],&ambient[a],&ambScale[a]);
      if((uintptr_t)hist[a]<begin || (uintptr_t)(hist[a]+NB)>end ||
         (uintptr_t)scale[a]<begin || (uintptr_t)(scale[a]+NB)>end ||
         (uintptr_t)ambient[a]<begin || (uintptr_t)(ambient[a]+1)>end ||
         (uintptr_t)ambScale[a]<begin || (uintptr_t)(ambScale[a]+1)>end) {
        cnhOutput.println("{\"type\":\"error\",\"message\":\"cnh_bounds\"}"); running=false;return;
      }
    }
    lastFrame=millis(); ++seq;
    cnhOutput.printf("{\"type\":\"cnh_frame\",\"seq\":%lu,\"ms\":%lu,\"read_us\":%lu,\"stream\":%u,\"rows\":8,\"cols\":8,\"bins\":%d,\"distance_mm\":[",seq,lastFrame,readUs,dev.streamcount,NB);
    for(int i=0;i<Z;++i){if(i)cnhOutput.print(',');cnhOutput.print(result.distance_mm[i]);}
    cnhOutput.print("],\"target_status\":[");
    for(int i=0;i<Z;++i){if(i)cnhOutput.print(',');cnhOutput.print(result.target_status[i]);}
    cnhOutput.print("],\"nb_target\":[");
    for(int i=0;i<Z;++i){if(i)cnhOutput.print(',');cnhOutput.print(result.nb_target_detected[i]);}
    cnhOutput.print("],\"hist_raw\":[");
    for(int a=0;a<Z;++a){if(a)cnhOutput.print(',');cnhOutput.print('[');for(int b=0;b<NB;++b){if(b)cnhOutput.print(',');cnhOutput.print(hist[a][b]);}cnhOutput.print(']');}
    cnhOutput.print("],\"hist_scaler\":[");
    for(int a=0;a<Z;++a){if(a)cnhOutput.print(',');cnhOutput.print('[');for(int b=0;b<NB;++b){if(b)cnhOutput.print(',');cnhOutput.print(scale[a][b]);}cnhOutput.print(']');}
    cnhOutput.print("],\"ambient_raw\":[");
    for(int a=0;a<Z;++a){if(a)cnhOutput.print(',');cnhOutput.print(*ambient[a]);}
    cnhOutput.print("],\"ambient_scaler\":[");
    for(int a=0;a<Z;++a){if(a)cnhOutput.print(',');cnhOutput.print(*ambScale[a]);}
    cnhOutput.print("],\"cnh_header\":[");
    for(int i=0;i<5;++i){if(i)cnhOutput.print(',');cnhOutput.print(cnh[i]);}
    cnhOutput.println("]}");
  }
  if(millis()-lastFrame>3000 && millis()-lastNote>3000){lastNote=millis();cnhOutput.println("{\"type\":\"error\",\"message\":\"no_fresh_frame\"}");}
  delay(2);
}
