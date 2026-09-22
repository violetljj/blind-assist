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
  if(!check("frequency_5hz",vl53lmz_set_ranging_frequency_hz(&device,5)))return false;
  return check("start",vl53lmz_start_ranging(&device));
}
void setup() {
  Serial.begin(115200);delay(1500);
  beginDemoNetwork("tof","xiao-tof-8x8-wifi-v1");
  api.on("/api/tof",HTTP_GET,[]{
    api.sendHeader("Cache-Control","no-store");
    if(!sequence){api.send(503,"application/json","{\"error\":\"no_frame\"}");return;}
    char times[220];snprintf(times,sizeof(times),"\",\"seq\":%lu,\"sampled_us\":%lld,\"send_us\":%lld,\"rows\":8,\"cols\":8,",
      (unsigned long)sequence,(long long)sampledUs,(long long)esp_timer_get_time());
    api.send(200,"application/json",String("{\"boot_id\":\"")+bootId+times+arrays+"}");
  });
  running=startSensor();lastInitAttempt=millis();
}
void loop() {
  demoNetworkLoop();
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
    }
  }
  delay(1);
}
