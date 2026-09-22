#pragma once
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <WebServer.h>
#include <Preferences.h>
#include "esp_timer.h"

// Shared local-network transport. Credentials are accepted over physical USB only.
static WebServer api(80);
static WiFiUDP clockUdp, discoveryUdp;
static String bootId, deviceId;
static const char* demoRole;
static const char* demoFirmware;
static char provisioning[160];
static size_t provisioningUsed = 0;
static bool provisioningOverflow = false, networkWasReady = false;
static uint32_t reconnectAt = 0;

static String demoIdentity() {
  return String("{\"role\":\"")+demoRole+"\",\"device_id\":\""+deviceId+
    "\",\"boot_id\":\""+bootId+"\",\"firmware\":\""+demoFirmware+
    "\",\"endpoint\":\"http://"+WiFi.localIP().toString()+"\"}";
}
static void beginDemoNetwork(const char* role, const char* firmware) {
  demoRole=role; demoFirmware=firmware;
  char id[17]; snprintf(id,sizeof(id),"%08lx%08lx",(unsigned long)esp_random(),(unsigned long)esp_random());
  bootId=id;
  WiFi.mode(WIFI_STA); WiFi.setSleep(false); WiFi.setAutoReconnect(true);
  deviceId=WiFi.macAddress();
  Preferences prefs; prefs.begin("ba_demo_wifi",true);
  String ssid=prefs.getString("ssid",""); String password=prefs.getString("pass",""); prefs.end();
  // Existing Atom configuration may be reused without exposing it or erasing NVS.
  if(ssid.isEmpty() && String(role)=="camera") {
    prefs.begin("tof4m_wifi",true); ssid=prefs.getString("ssid",""); password=prefs.getString("password",""); prefs.end();
  }
  if(!ssid.isEmpty()) WiFi.begin(ssid.c_str(),password.c_str());
  Serial.printf("{\"type\":\"network_config\",\"configured\":%s}\n",ssid.isEmpty()?"false":"true");
  api.on("/api/status",HTTP_GET,[]{api.sendHeader("Cache-Control","no-store");api.send(200,"application/json",demoIdentity());});
  api.begin();
}
static void demoNetworkLoop() {
  // WIFI<TAB>ssid<TAB>password<LF>; never echo credentials or malformed input.
  while(Serial.available()) {
    char c=Serial.read();
    if(c=='\n') {
      provisioning[provisioningUsed]=0;
      if(!provisioningOverflow && strncmp(provisioning,"WIFI\t",5)==0) {
        char* ssid=provisioning+5; char* password=strchr(ssid,'\t');
        if(password) {
          *password++=0;
          if(strlen(ssid)>0 && strlen(ssid)<=32 && strlen(password)>=8 && strlen(password)<=63 && !strchr(password,'\t')) {
            Preferences prefs; prefs.begin("ba_demo_wifi",false);
            prefs.putString("ssid",ssid);prefs.putString("pass",password);prefs.end();
            WiFi.disconnect();WiFi.begin(ssid,password);
            Serial.println("{\"type\":\"wifi_config_saved\"}");
          } else Serial.println("{\"type\":\"wifi_config_rejected\"}");
        }
      }
      memset(provisioning,0,sizeof(provisioning));provisioningUsed=0;provisioningOverflow=false;
    } else if(c!='\r') {
      if(provisioningUsed<sizeof(provisioning)-1) provisioning[provisioningUsed++]=c;
      else provisioningOverflow=true;
    }
  }
  bool connected=WiFi.status()==WL_CONNECTED;
  if(connected && !networkWasReady) {
    clockUdp.begin(3333);discoveryUdp.begin(3334);
    Serial.println(String("{\"type\":\"network_ready\",\"ip\":\"")+WiFi.localIP().toString()+"\",\"role\":\""+demoRole+"\"}");
  }
  if(!connected && networkWasReady) {clockUdp.stop();discoveryUdp.stop();}
  networkWasReady=connected;
  if(!connected) {
    if(millis()-reconnectAt>5000){reconnectAt=millis();WiFi.reconnect();}
    return;
  }
  int n=clockUdp.parsePacket();
  if(n) {
    int64_t received=esp_timer_get_time(); uint8_t req[16];
    int read=clockUdp.read(req,sizeof(req));
    if(n==16 && read==16 && memcmp(req,"BAT0",4)==0) {
      uint8_t response[24];memcpy(response,"BAT1",4);memcpy(response+4,req+4,4);
      memcpy(response+8,&received,8);int64_t sent=esp_timer_get_time();memcpy(response+16,&sent,8);
      clockUdp.beginPacket(clockUdp.remoteIP(),clockUdp.remotePort());clockUdp.write(response,sizeof(response));clockUdp.endPacket();
    }
  }
  n=discoveryUdp.parsePacket();
  if(n) {
    char req[64]={};int read=discoveryUdp.read(req,sizeof(req)-1);
    if(read==18 && n==18 && memcmp(req,"BADEMO_DISCOVER_V1",18)==0) {
      String response=demoIdentity();discoveryUdp.beginPacket(discoveryUdp.remoteIP(),discoveryUdp.remotePort());
      discoveryUdp.write((const uint8_t*)response.c_str(),response.length());discoveryUdp.endPacket();
    }
  }
  api.handleClient();
}
