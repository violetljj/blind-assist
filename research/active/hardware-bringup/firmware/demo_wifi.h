#pragma once
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <WebServer.h>
#include <Preferences.h>
#include "esp_timer.h"
#include "esp_mac.h"

// Shared local-network transport. Credentials are accepted over physical USB only.
static WebServer api(80);
static WiFiUDP clockUdp, discoveryUdp;
static String bootId, deviceId;
static const char* demoRole;
static const char* demoFirmware;
static char provisioning[224];
static size_t provisioningUsed = 0;
static bool provisioningOverflow = false, networkWasReady = false;
struct DemoWifiProfile { String ssid, password; };
static DemoWifiProfile wifiProfiles[2];
// One NVS key commits both profiles together. Original one-network keys remain for rollback.
struct DemoWifiStoredProfiles {
  uint32_t schema;
  char ssid[2][33], password[2][64];
};
static int wifiActiveProfile=-1, wifiPreferredProfile=0, wifiNextProfile=0;
static uint32_t wifiAttemptStarted=0, wifiBackoffStarted=0, wifiBackoffMs=0;
static uint32_t networkGeneration=0;
static uint8_t wifiFailures=0;
static bool wifiAttempting=false;
static const uint32_t wifiAttemptTimeoutMs=12000;

static bool validDemoCredential(const char* ssid, const char* password) {
  size_t s=strlen(ssid), p=strlen(password);
  return s>0 && s<=32 && p>=8 && p<=63 &&
    !strpbrk(ssid,"\t\r\n") && !strpbrk(password,"\t\r\n");
}
static unsigned demoProfileCount() {
  return (!wifiProfiles[0].ssid.isEmpty())+(!wifiProfiles[1].ssid.isEmpty());
}
static int nextDemoProfile(int preferred) {
  if(preferred>=0 && preferred<2 && !wifiProfiles[preferred].ssid.isEmpty())return preferred;
  return !wifiProfiles[0].ssid.isEmpty()?0:(!wifiProfiles[1].ssid.isEmpty()?1:-1);
}
static const char* demoNetworkState() {
  if(networkWasReady)return "connected";
  if(!demoProfileCount())return "unconfigured";
  return wifiAttempting?"connecting":"backoff";
}
static String demoNetworkMetadata() {
  return String("\"network\":{\"state\":\"")+demoNetworkState()+
    "\",\"profile_count\":"+String(demoProfileCount())+
    ",\"profile_index\":"+String(wifiActiveProfile)+
    ",\"generation\":"+String(networkGeneration)+
    ",\"attempt_timeout_ms\":12000,\"physical_boot\":false}";
}
static void demoNetworkDown(const char* reason) {
  if(!networkWasReady)return;
  clockUdp.stop();discoveryUdp.stop();networkWasReady=false;
  Serial.println(String("{\"type\":\"network_disconnected\",\"reason\":\"")+reason+
    "\",\"boot_id\":\""+bootId+"\","+demoNetworkMetadata()+"}");
}
static void startDemoWifiAttempt(int profile) {
  profile=nextDemoProfile(profile);
  if(profile<0)return;
  wifiActiveProfile=profile;wifiNextProfile=1-profile;
  WiFi.disconnect(false,false);
  WiFi.begin(wifiProfiles[profile].ssid.c_str(),wifiProfiles[profile].password.c_str());
  wifiAttemptStarted=millis();wifiAttempting=true;
  Serial.println(String("{\"type\":\"network_connect_attempt\",\"boot_id\":\"")+bootId+
    "\","+demoNetworkMetadata()+"}");
}
static void retryDemoWifi(const char* reason) {
  demoNetworkDown(reason);WiFi.disconnect(false,false);wifiAttempting=false;
  wifiNextProfile=nextDemoProfile(1-wifiActiveProfile);
  wifiBackoffStarted=millis();wifiBackoffMs=1000;
}
static bool saveDemoProfiles() {
  DemoWifiStoredProfiles stored={};stored.schema=1;
  for(int i=0;i<2;++i) {
    wifiProfiles[i].ssid.toCharArray(stored.ssid[i],sizeof(stored.ssid[i]));
    wifiProfiles[i].password.toCharArray(stored.password[i],sizeof(stored.password[i]));
  }
  Preferences prefs;
  if(!prefs.begin("ba_demo_wifi",false))return false;
  bool saved=prefs.putBytes("profiles2",&stored,sizeof(stored))==sizeof(stored);
  prefs.end();return saved;
}
static void loadDemoProfiles(const char* role) {
  Preferences prefs;prefs.begin("ba_demo_wifi",true);
  DemoWifiStoredProfiles stored={};
  bool loaded=prefs.getBytesLength("profiles2")==sizeof(stored) &&
    prefs.getBytes("profiles2",&stored,sizeof(stored))==sizeof(stored) && stored.schema==1;
  for(int i=0;i<2 && loaded;++i) {
    loaded=memchr(stored.ssid[i],0,sizeof(stored.ssid[i])) &&
      memchr(stored.password[i],0,sizeof(stored.password[i])) &&
      ((stored.ssid[i][0]==0 && stored.password[i][0]==0) ||
       validDemoCredential(stored.ssid[i],stored.password[i]));
  }
  if(loaded) {
    for(int i=0;i<2;++i){wifiProfiles[i].ssid=stored.ssid[i];wifiProfiles[i].password=stored.password[i];}
  } else {
    wifiProfiles[0].ssid=prefs.getString("ssid","");
    wifiProfiles[0].password=prefs.getString("pass","");
  }
  wifiPreferredProfile=prefs.getUChar("last_profile",0);prefs.end();
  // Existing Atom credentials are reused; no readout of their values or NVS erase.
  if(wifiProfiles[0].ssid.isEmpty() && String(role)=="camera") {
    prefs.begin("tof4m_wifi",true);wifiProfiles[0].ssid=prefs.getString("ssid","");
    wifiProfiles[0].password=prefs.getString("password","");prefs.end();
  }
  for(int i=0;i<2;++i) {
    if(!validDemoCredential(wifiProfiles[i].ssid.c_str(),wifiProfiles[i].password.c_str())) {
      wifiProfiles[i].ssid="";wifiProfiles[i].password="";
    }
  }
}

static void acceptDemoWifiCommand() {
  // Split in place, never echo fields, and reject controls/overflow before any write.
  if(provisioningOverflow){Serial.println("{\"type\":\"wifi_config_rejected\"}");return;}
  char* fields[5]={provisioning};unsigned count=1;
  for(char* c=provisioning;*c;++c)if(*c=='\t') {
    if(count==5){Serial.println("{\"type\":\"wifi_config_rejected\"}");return;}
    *c=0;fields[count++]=c+1;
  }
  if(count==1 && strcmp(fields[0],"WIFI_RETRY")==0) {
    retryDemoWifi("usb_retry");
    Serial.println("{\"type\":\"network_retry_requested\",\"physical_boot\":false}");return;
  }
  DemoWifiProfile previous[2]={wifiProfiles[0],wifiProfiles[1]};
  bool valid=false;int requested=-1;
  if(count==3 && strcmp(fields[0],"WIFI")==0 && validDemoCredential(fields[1],fields[2])) {
    wifiProfiles[0]={fields[1],fields[2]};valid=true;requested=0;
  } else if(count==4 && strcmp(fields[0],"WIFI_SLOT")==0 &&
            (strcmp(fields[1],"0")==0 || strcmp(fields[1],"1")==0) &&
            validDemoCredential(fields[2],fields[3])) {
    requested=fields[1][0]-'0';wifiProfiles[requested]={fields[2],fields[3]};valid=true;
  } else if(count==5 && strcmp(fields[0],"WIFI2")==0 &&
            validDemoCredential(fields[1],fields[2]) && validDemoCredential(fields[3],fields[4]) &&
            strcmp(fields[1],fields[3])!=0) {
    wifiProfiles[0]={fields[1],fields[2]};wifiProfiles[1]={fields[3],fields[4]};valid=true;
  }
  if(valid && wifiProfiles[0].ssid==wifiProfiles[1].ssid && !wifiProfiles[0].ssid.isEmpty())valid=false;
  if(!valid || !saveDemoProfiles()) {
    wifiProfiles[0]=previous[0];wifiProfiles[1]=previous[1];
    Serial.println("{\"type\":\"wifi_config_rejected\"}");return;
  }
  Serial.println(String("{\"type\":\"wifi_config_saved\",\"profile_count\":")+
    String(demoProfileCount())+",\"profile_index\":"+String(requested)+"}");
  bool keepCurrent=networkWasReady && wifiActiveProfile>=0 &&
    previous[wifiActiveProfile].ssid==wifiProfiles[wifiActiveProfile].ssid &&
    previous[wifiActiveProfile].password==wifiProfiles[wifiActiveProfile].password;
  if(keepCurrent) {
    // Adding the hotspot does not disturb a live LAN capture. A receipt confirms its current IP.
    Serial.println(String("{\"type\":\"network_ready\",\"ip\":\"")+WiFi.localIP().toString()+
      "\",\"role\":\""+demoRole+"\",\"boot_id\":\""+bootId+"\","+demoNetworkMetadata()+"}");
    return;
  }
  demoNetworkDown("configuration_changed");wifiFailures=0;
  startDemoWifiAttempt(requested>=0?requested:wifiPreferredProfile);
}

static String demoIdentity() {
  return String("{\"role\":\"")+demoRole+"\",\"device_id\":\""+deviceId+
    "\",\"boot_id\":\""+bootId+"\",\"firmware\":\""+demoFirmware+
    "\",\"endpoint\":\"http://"+WiFi.localIP().toString()+"\","+demoNetworkMetadata()+"}";
}
static void beginDemoNetwork(const char* role, const char* firmware) {
  demoRole=role; demoFirmware=firmware;
  char id[17]; snprintf(id,sizeof(id),"%08lx%08lx",(unsigned long)esp_random(),(unsigned long)esp_random());
  bootId=id;
  WiFi.mode(WIFI_STA); WiFi.setSleep(false); WiFi.setAutoReconnect(false);
  // WiFi.macAddress() can still be all-zero before the STA interface starts.
  // Read the SDK's stable station MAC directly; the eFuse base is a fallback.
  uint8_t stationMac[6]={};
  if(esp_read_mac(stationMac,ESP_MAC_WIFI_STA)!=ESP_OK &&
     esp_efuse_mac_get_default(stationMac)!=ESP_OK) {
    deviceId="unavailable";
    Serial.println("{\"type\":\"network_identity_error\"}");
  } else {
    char macId[18];snprintf(macId,sizeof(macId),"%02x:%02x:%02x:%02x:%02x:%02x",
      stationMac[0],stationMac[1],stationMac[2],stationMac[3],stationMac[4],stationMac[5]);
    deviceId=macId;
  }
  loadDemoProfiles(role);
  Serial.printf("{\"type\":\"network_config\",\"configured\":%s,\"profile_count\":%u}\n",
    demoProfileCount()?"true":"false",demoProfileCount());
  startDemoWifiAttempt(wifiPreferredProfile);
  api.on("/api/status",HTTP_GET,[]{api.sendHeader("Cache-Control","no-store");api.send(200,"application/json",demoIdentity());});
  api.begin();
}
static void demoNetworkLoop() {
  // USB-only WIFI, WIFI_SLOT and WIFI2 provisioning; never echo credential fields.
  while(Serial.available()) {
    char c=Serial.read();
    if(c=='\n') {
      provisioning[provisioningUsed]=0;
      acceptDemoWifiCommand();
      memset(provisioning,0,sizeof(provisioning));provisioningUsed=0;provisioningOverflow=false;
    } else {
      if(c=='\r' || c==0)provisioningOverflow=true;
      if(provisioningUsed<sizeof(provisioning)-1) provisioning[provisioningUsed++]=c;
      else provisioningOverflow=true;
    }
  }
  bool connected=WiFi.status()==WL_CONNECTED;
  if(connected && !networkWasReady) {
    wifiAttempting=false;wifiFailures=0;++networkGeneration;
    wifiPreferredProfile=wifiActiveProfile;
    Preferences prefs;if(prefs.begin("ba_demo_wifi",false)) {
      if(prefs.getUChar("last_profile",255)!=wifiPreferredProfile)prefs.putUChar("last_profile",wifiPreferredProfile);
      prefs.end();
    }
    clockUdp.begin(3333);discoveryUdp.begin(3334);
    networkWasReady=true;
    Serial.println(String("{\"type\":\"network_ready\",\"ip\":\"")+WiFi.localIP().toString()+
      "\",\"role\":\""+demoRole+"\",\"boot_id\":\""+bootId+"\","+demoNetworkMetadata()+"}");
  }
  if(!connected && networkWasReady) {
    retryDemoWifi("link_lost");
  }
  if(!connected) {
    uint32_t now=millis();
    if(wifiAttempting && (uint32_t)(now-wifiAttemptStarted)>=wifiAttemptTimeoutMs) {
      WiFi.disconnect(false,false);wifiAttempting=false;
      wifiBackoffStarted=now;wifiBackoffMs=1000u<<min((unsigned)wifiFailures,3u);
      if(wifiFailures<3)++wifiFailures;
      Serial.println(String("{\"type\":\"network_attempt_timeout\",\"boot_id\":\"")+bootId+
        "\",\"retry_in_ms\":"+String(wifiBackoffMs)+","+demoNetworkMetadata()+"}");
    }
    if(!wifiAttempting && demoProfileCount() && (uint32_t)(now-wifiBackoffStarted)>=wifiBackoffMs)
      startDemoWifiAttempt(wifiNextProfile);
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
