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
static bool probeSensorAtStartup();
static String cnhHttpStatus();
static void serviceCnhHttp();
static void prepareHttpResponse(unsigned code, const String& payload);
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
// WiFi/phone storage can briefly pause reads. Retain only the current line and
// tolerate a bounded pause; timestamps/sequence gaps still expose the delay.
static constexpr uint32_t CNH_TCP_ROW_DEADLINE_US = 5000000;
static constexpr uint32_t CNH_TCP_STALL_DEADLINE_US = 2000000;
static uint32_t cnhSendDisconnects = 0, cnhSendTimeouts = 0;
static uint32_t cnhLastSuccessfulSendUs = 0, cnhMaxSuccessfulSendUs = 0;
static uint32_t networkLoopLastUs = 0, networkLoopMaxUs = 0;
static uint32_t sensorLastFrameUs = 0, sensorMaxIntervalUs = 0;
static const char* cnhLastDisconnectReason = "none";
static uint32_t cnhLastDisconnectSent = 0, cnhLastDisconnectTotal = 0;
static uint32_t cnhLastDisconnectElapsedUs = 0;
static int cnhLastDisconnectErrno = 0;
static bool sendCnhClient(const String& text) {
  if (!cnhClient.connected()) return false;
  const uint8_t* data = (const uint8_t*)text.c_str();
  size_t sent = 0;
  const uint32_t started = micros();
  uint32_t lastProgress = started;
  const char* failure = nullptr;
  int failureErrno = 0;
  while (sent < text.length()) {
    const uint32_t now = micros();
    if ((uint32_t)(now-started) >= CNH_TCP_ROW_DEADLINE_US) {
      failure = "row_timeout"; ++cnhSendTimeouts; break;
    }
    if ((uint32_t)(now-lastProgress) >= CNH_TCP_STALL_DEADLINE_US) {
      failure = "stall_timeout"; ++cnhSendTimeouts; break;
    }
    int count = send(cnhClient.fd(), data+sent, text.length()-sent, MSG_DONTWAIT);
    if (count > 0) {
      sent += count;
      lastProgress = micros();
    }
    else if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      failure = "send_error"; failureErrno = errno; break;
    }
    else delay(1);
  }
  if (sent != text.length()) {
    ++cnhSendDisconnects;
    cnhLastDisconnectReason = failure ? failure : "incomplete_send";
    cnhLastDisconnectSent = sent;
    cnhLastDisconnectTotal = text.length();
    cnhLastDisconnectElapsedUs = (uint32_t)(micros()-started);
    cnhLastDisconnectErrno = failureErrno;
    cnhClient.stop();
    return false;
  }
  cnhLastSuccessfulSendUs = (uint32_t)(micros()-started);
  cnhMaxSuccessfulSendUs = max(cnhMaxSuccessfulSendUs, cnhLastSuccessfulSendUs);
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
static uint32_t bytes=0, seq=0, lastFrame=0, lastNote=0, readUs=0, i2cReadUs=0;
static bool running=false;
static const char* lastOp="none";
static unsigned lastStatus=0;
static uint8_t startupProbeAttempts=0, startupFirstStatus=0, startupFirstAlive=0;
static bool startupBusRecovery=false;
static int startupSdaBefore=-1, startupSdaAfter=-1, startupSclAfter=-1;
static bool ok(const char* op, uint8_t s) {
  cnhOutput.printf("{\"type\":\"step\",\"op\":\"%s\",\"status\":%u}\n",op,s);
  lastOp=op; lastStatus=s;
  return s==0;
}
static bool probeSensorAtStartup() {
  uint8_t alive=0;
  startupProbeAttempts=1;
  startupFirstStatus=vl53lmz_is_alive(&dev,&alive);
  startupFirstAlive=alive;
  if (ok("is_alive",startupFirstStatus) && alive) return true;
  // The sensor remains powered across an MCU reset, which can interrupt an
  // I2C read with SDA held low. Match tof_wifi's bus clear before ONE retry;
  // never reset or reconfigure the sensor during an active capture.
  startupBusRecovery=true;
  Wire.end();
  pinMode(5,INPUT_PULLUP);
  pinMode(6,OUTPUT_OPEN_DRAIN); digitalWrite(6,HIGH);
  delayMicroseconds(10);
  startupSdaBefore=digitalRead(5);
  for(int i=0;i<9;++i) {
    digitalWrite(6,LOW); delayMicroseconds(10);
    digitalWrite(6,HIGH); delayMicroseconds(10);
  }
  // STOP: release SDA while SCL is high, using open drain throughout.
  digitalWrite(6,LOW);
  pinMode(5,OUTPUT_OPEN_DRAIN); digitalWrite(5,LOW);
  delayMicroseconds(10);
  digitalWrite(6,HIGH); delayMicroseconds(10);
  digitalWrite(5,HIGH); delayMicroseconds(10);
  startupSdaAfter=digitalRead(5); startupSclAfter=digitalRead(6);
  cnhOutput.printf("{\"type\":\"startup_bus_recovery\",\"first_status\":%u,\"first_alive\":%u,\"sda_before\":%d,\"sda_after\":%d,\"scl_after\":%d}\n",
    startupFirstStatus,startupFirstAlive,startupSdaBefore,startupSdaAfter,startupSclAfter);
  if (!Wire.begin(5,6,CNH_I2C_HZ)) return ok("wire_reopen",1);
  Wire.setTimeOut(100);
  if (!ok("i2c_read_backend_reopen",UseEsp32HalRead(0))) return false;
  alive=0; startupProbeAttempts=2;
  return ok("is_alive_retry",vl53lmz_is_alive(&dev,&alive)) && alive;
}
// The SDK WebServer parser busy-spins on incomplete request lines and contends
// on the lwIP lock even from a separate task. H3 needs two read-only GET routes:
// collect headers incrementally, then send a fixed bounded response. No polling
// loop waits for another byte, and identity/configuration stays on this task.
static WiFiServer cnhHttpServer(80);
static WiFiClient cnhHttpClient;
static constexpr size_t HTTP_REQUEST_BYTES = 2048, HTTP_RESPONSE_BYTES = 4608;
static constexpr uint32_t HTTP_REQUEST_DEADLINE_MS = 5000;
static char httpRequest[HTTP_REQUEST_BYTES], httpResponse[HTTP_RESPONSE_BYTES];
static size_t httpRequestUsed = 0, httpResponseSize = 0, httpResponseSent = 0;
static uint32_t httpRequestStarted = 0;

static String cnhHttpStatus() {
  String status = demoIdentity();
  String identity=status; identity.remove(identity.length()-1);
  String payload=identity+",\"transport\":\"tcp_ndjson\",\"tcp_port\":3335,\"firmware_config\":";
  payload+=cnhOutput.configRecord.isEmpty()?"null":cnhOutput.configRecord;
  payload+=",\"startup_diagnostics\":{\"probe_attempts\":"+String(startupProbeAttempts)+
    ",\"first_status\":"+String(startupFirstStatus)+",\"first_alive\":"+String(startupFirstAlive)+
    ",\"bus_recovery\":"+(startupBusRecovery?"true":"false")+
    ",\"sda_before\":"+String(startupSdaBefore)+",\"sda_after\":"+String(startupSdaAfter)+
    ",\"scl_after\":"+String(startupSclAfter)+"}";
  payload+=",\"tcp_diagnostics\":{\"row_deadline_ms\":"+String(CNH_TCP_ROW_DEADLINE_US/1000)+
    ",\"stall_deadline_ms\":"+String(CNH_TCP_STALL_DEADLINE_US/1000)+
    ",\"send_disconnect_count\":"+String(cnhSendDisconnects)+
    ",\"send_timeout_count\":"+String(cnhSendTimeouts)+
    ",\"last_disconnect_reason\":\""+cnhLastDisconnectReason+
    "\",\"last_sent_bytes\":"+String(cnhLastDisconnectSent)+
    ",\"last_total_bytes\":"+String(cnhLastDisconnectTotal)+
    ",\"last_elapsed_us\":"+String(cnhLastDisconnectElapsedUs)+
    ",\"last_errno\":"+String(cnhLastDisconnectErrno)+
      ",\"last_successful_send_us\":"+String(cnhLastSuccessfulSendUs)+
      ",\"max_successful_send_us\":"+String(cnhMaxSuccessfulSendUs)+
      ",\"network_loop_last_us\":"+String(networkLoopLastUs)+
      ",\"network_loop_max_us\":"+String(networkLoopMaxUs)+
      ",\"sensor_frame_interval_max_us\":"+String(sensorMaxIntervalUs)+"}";
  payload+=",\"snapshot_ms\":"+String(millis())+",\"http_service\":\"bounded_poll\"}";
  return payload;
}

static void prepareHttpResponse(unsigned code, const String& payload) {
  const char* body = payload.c_str();
  size_t bodySize = payload.length();
  if (bodySize > HTTP_RESPONSE_BYTES-256) {
    code = 503; body = "{\"error\":\"http_response_overflow\"}"; bodySize = strlen(body);
  }
  const char* reason = code==200 ? "OK" : code==404 ? "Not Found" :
      code==405 ? "Method Not Allowed" : code==431 ? "Request Header Fields Too Large" :
      code==503 ? "Service Unavailable" : "Bad Request";
  const int headerSize = snprintf(httpResponse, sizeof(httpResponse),
      "HTTP/1.1 %u %s\r\nContent-Type: application/json\r\nContent-Length: %u\r\n"
      "Cache-Control: no-store\r\nConnection: close\r\n\r\n", code, reason, (unsigned)bodySize);
  memcpy(httpResponse+headerSize, body, bodySize);
  httpResponseSize = headerSize+bodySize;
  httpResponseSent = 0;
}

static void serviceCnhHttp() {
  if (WiFi.status()!=WL_CONNECTED) { cnhHttpClient.stop(); return; }
  if (!cnhHttpClient) {
    cnhHttpClient = cnhHttpServer.accept();
    if (!cnhHttpClient) return;
    cnhHttpClient.setNoDelay(true);
    httpRequestUsed = httpResponseSize = httpResponseSent = 0;
    httpRequestStarted = millis();
  }
  if ((uint32_t)(millis()-httpRequestStarted)>=HTTP_REQUEST_DEADLINE_MS) {
    cnhHttpClient.stop(); return;
  }
  if (!httpResponseSize) {
    const size_t capacity = min((size_t)1024, sizeof(httpRequest)-1-httpRequestUsed);
    const int count = recv(cnhHttpClient.fd(), httpRequest+httpRequestUsed, capacity, MSG_DONTWAIT);
    if (count==0) { cnhHttpClient.stop(); return; }
    if (count<0) {
      if (errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR) cnhHttpClient.stop();
      return;
    }
    httpRequestUsed += count;
    httpRequest[httpRequestUsed] = 0;
    if (strstr(httpRequest, "\r\n\r\n")) {
      char* lineEnd = strstr(httpRequest, "\r\n");
      char* path = strchr(httpRequest, ' ');
      char* version = path ? strchr(path+1, ' ') : nullptr;
      if (!lineEnd || !path || !version || version>=lineEnd) {
        prepareHttpResponse(400, "{\"error\":\"bad_request\"}");
      } else if (strncmp(httpRequest, "GET ", 4)!=0) {
        prepareHttpResponse(405, "{\"error\":\"get_only\"}");
      } else {
        ++path; *version = 0;
        char* query = strchr(path, '?'); if (query) *query = 0;
        if (strcmp(path, "/api/status")==0) prepareHttpResponse(200, demoIdentity());
        else if (strcmp(path, "/api/cnh")==0) prepareHttpResponse(200, cnhHttpStatus());
        else prepareHttpResponse(404, "{\"error\":\"not_found\"}");
      }
    } else if (httpRequestUsed==sizeof(httpRequest)-1) {
      prepareHttpResponse(431, "{\"error\":\"http_request_too_large\"}");
    }
  }
  if (httpResponseSize) {
    const size_t count = min((size_t)1024, httpResponseSize-httpResponseSent);
    const int sent = send(cnhHttpClient.fd(), httpResponse+httpResponseSent, count, MSG_DONTWAIT);
    if (sent>0) {
      httpResponseSent += sent;
      if (httpResponseSent==httpResponseSize) cnhHttpClient.stop();
    } else if (sent<0 && errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR) cnhHttpClient.stop();
  }
}

void setup() {
  cnhOutput.begin(115200); delay(2000);
  beginDemoNetwork("tof","xiao-cnh-h3-wifi-v1");
  cnhServer.begin();
  // H3 replaces the synchronous WebServer with bounded read-only HTTP polling.
  api.stop();
  cnhHttpServer.begin();
  cnhOutput.printf("{\"type\":\"boot\",\"firmware\":\"xiao-cnh-h3-wifi-v1\",\"startup_kind\":\"physical_boot\",\"boot_id\":\"%s\",\"device_id\":\"%s\"}\n",bootId.c_str(),deviceId.c_str());
  if (!Wire.begin(5,6,CNH_I2C_HZ)) { ok("wire_begin",1); return; }
  Wire.setTimeOut(100);
  if (!ok("i2c_read_backend",UseEsp32HalRead(0))) return;
  dev.platform.address=0x52;
  if(!probeSensorAtStartup()) return;
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
  cnhOutput.printf("{\"type\":\"config\",\"zones\":%d,\"bins\":%d,\"start_bin\":0,\"sub_sample\":8,\"base_bin_mm\":37.5348,\"i2c_hz\":%lu,\"read_backend\":\"esp32_hal_direct\",\"read_timeout_policy\":\"max_100ms_2x_wire_ceil_ms_plus_25ms\",\"cnh_bytes\":%lu,\"read_bytes\":%lu,\"frequency_hz\":5,\"integration_ms\":20,\"ranging_mode\":\"autonomous\",\"transport\":\"tcp_ndjson\",\"tcp_port\":3335}\n",
                Z,NB,(unsigned long)CNH_I2C_HZ,bytes,dev.data_read_size);
  running=true; lastFrame=millis();
}
void loop() {
  const uint32_t networkStarted = micros();
  demoNetworkLoop(false,cnhClient.connected());
  networkLoopLastUs = (uint32_t)(micros()-networkStarted);
  networkLoopMaxUs = max(networkLoopMaxUs, networkLoopLastUs);
  serviceCnhHttp();
  serviceCnhClient();
  if(!running) {
    if (!lastNote || (uint32_t)(millis()-lastNote)>=3000) {
      lastNote=millis();
      cnhOutput.printf("{\"type\":\"error\",\"message\":\"init_failed\",\"op\":\"%s\",\"status\":%u,\"cnh_bytes\":%lu}\n",lastOp,lastStatus,bytes);
    }
    delay(2);return;
  }
  uint8_t ready=0, s=vl53lmz_check_data_ready(&dev,&ready);
  if(s) {ok("ready",s);delay(500);return;}
  if(ready) {
    const uint32_t transferBefore = ReadTransferUs();
    uint32_t t0=micros();
    s=vl53lmz_get_ranging_data(&dev,&result);
    readUs=micros()-t0;
    i2cReadUs=(uint32_t)(ReadTransferUs()-transferBefore);
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
    const uint32_t frameUs = micros();
    if (seq) sensorMaxIntervalUs = max(sensorMaxIntervalUs, (uint32_t)(frameUs-sensorLastFrameUs));
    sensorLastFrameUs = frameUs;
    lastFrame=millis(); ++seq;
    cnhOutput.printf("{\"type\":\"cnh_frame\",\"seq\":%lu,\"ms\":%lu,\"read_us\":%lu,\"i2c_read_us\":%lu,\"stream\":%u,\"rows\":8,\"cols\":8,\"bins\":%d,\"distance_mm\":[",seq,lastFrame,readUs,i2cReadUs,dev.streamcount,NB);
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
