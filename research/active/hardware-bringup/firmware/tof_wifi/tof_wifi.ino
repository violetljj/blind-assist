#include <Arduino.h>
#include <Wire.h>
#include "demo_wifi.h"
extern "C" {
#include "vl53lmz_api.h"
}
static VL53LMZ_Configuration device;
static VL53LMZ_ResultsData result;
static bool running=false;
static uint32_t sequence=0;
static int64_t sampledUs=0;
static String arrays;
static uint32_t lastInitAttempt=0;
static WiFiUDP sampleUdp;
static bool sampleUdpReady=false, subscriberActive=false;
static IPAddress subscriberIp;
static uint16_t subscriberPort=0;
static uint32_t subscribedAt=0;
static constexpr char kSubscribe[]="BADEMO_TOF_V1";
static constexpr uint32_t kLeaseMs=5000;
static String sampleJson() {
  char times[220];snprintf(times,sizeof(times),"\",\"seq\":%lu,\"sampled_us\":%lld,\"send_us\":%lld,\"rows\":8,\"cols\":8,",
    (unsigned long)sequence,(long long)sampledUs,(long long)esp_timer_get_time());
  return String("{\"boot_id\":\"")+bootId+times+arrays+"}";
}
static void serviceSubscription() {
  if(WiFi.status()!=WL_CONNECTED) {
    if(sampleUdpReady)sampleUdp.stop();
    sampleUdpReady=false;subscriberActive=false;return;
  }
  if(!sampleUdpReady)sampleUdpReady=sampleUdp.begin(3335)!=0;
  if(!sampleUdpReady)return;
  if(subscriberActive && (uint32_t)(millis()-subscribedAt)>=kLeaseMs)subscriberActive=false;
  // Bound control work so an incoming burst cannot indefinitely delay sampling.
  for(int i=0;i<4;++i) {
    int size=sampleUdp.parsePacket();if(!size)break;
    char request[32];int count=sampleUdp.read(request,sizeof(request));
    if(size==(int)sizeof(kSubscribe)-1 && count==size && memcmp(request,kSubscribe,size)==0) {
      subscriberIp=sampleUdp.remoteIP();subscriberPort=sampleUdp.remotePort();
      subscribedAt=millis();subscriberActive=true;
    }
  }
}
static void sendLatestSample() {
  if(!sampleUdpReady || !subscriberActive || (uint32_t)(millis()-subscribedAt)>=kLeaseMs)return;
  // Exactly one datagram for this newly acquired sample; no queue or old-frame replay.
  if(!sampleUdp.beginPacket(subscriberIp,subscriberPort))return;
  String payload=sampleJson();
  sampleUdp.write((const uint8_t*)payload.c_str(),payload.length());
  sampleUdp.endPacket();
}
static bool check(const char* op,uint8_t status) {
  Serial.printf("{\"type\":\"step\",\"op\":\"%s\",\"status\":%u}\n",op,status);return status==0;
}
static bool startSensor() {
  // An MCU reset can interrupt a sensor transaction with SDA held low.
  // Release the bus with nine clocks and STOP before reopening Wire.
  Wire.end();
  pinMode(5,INPUT_PULLUP);pinMode(6,OUTPUT_OPEN_DRAIN);digitalWrite(6,HIGH);
  for(int i=0;i<9;++i){digitalWrite(6,LOW);delayMicroseconds(10);digitalWrite(6,HIGH);delayMicroseconds(10);}
  pinMode(5,OUTPUT_OPEN_DRAIN);digitalWrite(5,LOW);delayMicroseconds(10);
  digitalWrite(6,HIGH);delayMicroseconds(10);digitalWrite(5,HIGH);delayMicroseconds(10);
  Wire.begin(5,6,400000);Wire.setTimeOut(100);device.platform.address=0x52;
  uint8_t alive=0;
  if(!check("is_alive",vl53lmz_is_alive(&device,&alive)) || !alive)return false;
  if(!check("init",vl53lmz_init(&device)))return false;
  if(!check("resolution_8x8",vl53lmz_set_resolution(&device,VL53LMZ_RESOLUTION_8X8)))return false;
  if(!check("frequency_10hz",vl53lmz_set_ranging_frequency_hz(&device,10)))return false;
  return check("start",vl53lmz_start_ranging(&device));
}
void setup() {
  Serial.begin(115200);delay(1500);
  beginDemoNetwork("tof","xiao-tof-8x8-wifi-v3");
  api.on("/api/tof",HTTP_GET,[]{
    api.client().setNoDelay(true);
    api.sendHeader("Cache-Control","no-store");
    if(!sequence){api.send(503,"application/json","{\"error\":\"no_frame\"}");return;}
    api.send(200,"application/json",sampleJson());
  });
  running=startSensor();lastInitAttempt=millis();
}
void loop() {
  demoNetworkLoop();
  serviceSubscription();
  if(!running && millis()-lastInitAttempt>5000){running=startSensor();lastInitAttempt=millis();}
  if(running) {
    uint8_t ready=0;
    if(vl53lmz_check_data_ready(&device,&ready)==0 && ready && vl53lmz_get_ranging_data(&device,&result)==0) {
      sampledUs=esp_timer_get_time();++sequence;
      arrays="\"distance_mm\":[";
      for(int i=0;i<64;++i){if(i)arrays+=',';arrays+=String(result.distance_mm[i]);}
      arrays+="],\"target_status\":[";
      for(int i=0;i<64;++i){if(i)arrays+=',';arrays+=String(result.target_status[i]);}
      arrays+="],\"nb_target\":[";
      for(int i=0;i<64;++i){if(i)arrays+=',';arrays+=String(result.nb_target_detected[i]);}
      arrays+=']';
      sendLatestSample();
    }
  }
  delay(1);
}
