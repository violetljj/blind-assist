// Host mocks exercise the actual shared header. Synthetic behavior, not device evidence.
#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>
using std::min;
class String : public std::string {
 public:
  using std::string::string;
  String()=default;
  String(const std::string& value):std::string(value){}
  String(int value):std::string(std::to_string(value)){}
  String(unsigned value):std::string(std::to_string(value)){}
  bool isEmpty() const{return empty();}
  void toCharArray(char* out,size_t size)const{std::snprintf(out,size,"%s",c_str());}
};
static uint32_t fakeNow=0;
uint32_t millis(){return fakeNow;}
uint32_t esp_random(){return 42;}
int64_t esp_timer_get_time(){return (int64_t)fakeNow*1000;}
static constexpr int ESP_OK=0,ESP_MAC_WIFI_STA=0;
int esp_read_mac(uint8_t* mac,int){const uint8_t fixed[6]={0xb4,0x3a,0x45,0xbd,0x12,0xd8};memcpy(mac,fixed,6);return ESP_OK;}
int esp_efuse_mac_get_default(uint8_t* mac){return esp_read_mac(mac,ESP_MAC_WIFI_STA);}
class MockSerial {
 public:
  std::deque<char> incoming;
  std::string output;
  int available(){return (int)incoming.size();}
  char read(){char c=incoming.front();incoming.pop_front();return c;}
  void println(const String& value){output+=value;output+='\n';}
  void printf(const char* format,...){char value[512];va_list args;va_start(args,format);vsnprintf(value,sizeof(value),format,args);va_end(args);output+=value;}
  void command(const std::string& value){for(char c:value)incoming.push_back(c);}
} Serial;
class IPAddress {
 public:
  String toString()const{return "192.168.5.10";}
};
static constexpr int WL_CONNECTED=3,WIFI_STA=1,HTTP_GET=0,WIFI_SCAN_RUNNING=-1,WIFI_SCAN_FAILED=-2;
class MockWiFi {
 public:
  bool connected=false,automatic=true;
  unsigned starts=0,disconnects=0;
  unsigned scans=0,scanDeletes=0,scanStops=0;
  uint32_t scanTimeout=0;
  int scanStartResult=WIFI_SCAN_RUNNING,scanResult=WIFI_SCAN_FAILED;
  std::vector<String> scanResults;
  String attempted,scanFilter;
  void mode(int){}
  void setSleep(bool){}
  void setAutoReconnect(bool value){automatic=value;}
  String macAddress(){return "00:00:00:00:00:00";}
  IPAddress localIP(){return {};}
  int status(){return connected?WL_CONNECTED:0;}
  void disconnect(bool=false,bool=false){connected=false;++disconnects;}
  void begin(const char* ssid,const char*){attempted=ssid;++starts;}
  void setScanTimeout(uint32_t value){scanTimeout=value;}
  int scanNetworks(bool async,bool hidden,bool passive,uint32_t dwell,uint8_t channel,const char* ssid){
    assert(async && !hidden && !passive && dwell==120 && channel==0);
    scanFilter=ssid;++scans;scanResult=scanStartResult;return scanStartResult;
  }
  int scanComplete(){return scanResult;}
  String SSID(int index){return scanResults.at(index);}
  void scanDelete(){++scanDeletes;scanResults.clear();scanResult=WIFI_SCAN_FAILED;}
  void completeScan(std::vector<String> results){scanResults=results;scanResult=(int)results.size();}
} WiFi;
int esp_wifi_scan_stop(){++WiFi.scanStops;WiFi.scanResult=WIFI_SCAN_FAILED;return ESP_OK;}
class WiFiUDP {
 public:
  unsigned starts=0,stops=0;
  int begin(unsigned){++starts;return 1;}
  void stop(){++stops;}
  int parsePacket(){return 0;}
  int read(void*,size_t){return 0;}
  IPAddress remoteIP(){return {};}
  unsigned remotePort(){return 0;}
  void beginPacket(IPAddress,unsigned){}
  void write(const uint8_t*,size_t){}
  void endPacket(){}
};
class WebServer {
 public:
  unsigned handled=0;
  explicit WebServer(int){}
  template<class F>void on(const char*,int,F){}
  void sendHeader(const char*,const char*){}
  void send(int,const char*,const String&){}
  void begin(){}
  void handleClient(){++handled;}
};
class Preferences {
  std::string name;
 public:
  static std::map<std::string,String> strings;
  static std::map<std::string,std::vector<uint8_t>> blobs;
  static std::map<std::string,uint8_t> numbers;
  static bool rejectWrite;
  bool begin(const char* value,bool){name=value;return true;}
  void end(){}
  std::string key(const char* value)const{return name+"/"+value;}
  String getString(const char* value,const char* fallback){auto i=strings.find(key(value));return i==strings.end()?String(fallback):i->second;}
  uint8_t getUChar(const char* value,uint8_t fallback){auto i=numbers.find(key(value));return i==numbers.end()?fallback:i->second;}
  size_t putUChar(const char* value,uint8_t number){numbers[key(value)]=number;return 1;}
  size_t getBytesLength(const char* value){return blobs[key(value)].size();}
  size_t getBytes(const char* value,void* out,size_t length){auto& b=blobs[key(value)];if(b.size()!=length)return 0;memcpy(out,b.data(),length);return length;}
  size_t putBytes(const char* value,const void* in,size_t length){if(rejectWrite)return 0;auto b=(const uint8_t*)in;blobs[key(value)]=std::vector<uint8_t>(b,b+length);return length;}
};
std::map<std::string,String> Preferences::strings;
std::map<std::string,std::vector<uint8_t>> Preferences::blobs;
std::map<std::string,uint8_t> Preferences::numbers;
bool Preferences::rejectWrite=false;
#include "../firmware/demo_wifi.h"

static void tick(uint32_t delta=0){fakeNow+=delta;demoNetworkLoop();}
static void resetRuntime(bool clearStored=true){
  fakeNow=0;Serial={};WiFi={};clockUdp={};discoveryUdp={};api=WebServer(80);
  provisioningUsed=0;provisioningOverflow=false;memset(provisioning,0,sizeof(provisioning));
  networkWasReady=false;wifiProfiles[0]={};wifiProfiles[1]={};
  wifiActiveProfile=-1;wifiNextProfile=1;wifiAttemptStarted=0;
  wifiBackoffStarted=0;wifiBackoffMs=0;wifiFailures=0;wifiAttempting=false;
  wifiHotspotScanning=false;wifiHotspotCheckStarted=0;wifiHotspotScanStarted=0;
  wifiHotspotCheckDelayMs=wifiHotspotScanIntervalMs;networkGeneration=0;
  if(clearStored){Preferences::strings.clear();Preferences::blobs.clear();Preferences::numbers.clear();}
  Preferences::rejectWrite=false;
}
static void checkNoCredentials(){
  assert(Serial.output.find("secret-")==std::string::npos);
  assert(demoIdentity().find("secret-")==std::string::npos);
  assert(Serial.output.find("synthetic-lan")==std::string::npos);
  assert(Serial.output.find("synthetic-hotspot")==std::string::npos);
}
int main(){
  resetRuntime();
  Preferences::strings["ba_demo_wifi/ssid"]="synthetic-lan";
  Preferences::strings["ba_demo_wifi/pass"]="secret-lan-password";
  beginDemoNetwork("tof","synthetic-firmware");
  const String physicalBoot=bootId;
  assert(deviceId=="b4:3a:45:bd:12:d8");
  assert(WiFi.attempted=="synthetic-lan" && !WiFi.automatic);
  WiFi.connected=true;tick();
  assert(networkGeneration==1 && networkWasReady);
  const unsigned connectedStarts=WiFi.starts,connectedDisconnects=WiFi.disconnects;
  Serial.command("WIFI_SLOT\t1\tsynthetic-hotspot\tsecret-hotspot-password\n");tick();
  assert(demoProfileCount()==2 && WiFi.starts==connectedStarts && WiFi.disconnects==connectedDisconnects);
  assert(Preferences::strings["ba_demo_wifi/ssid"]=="synthetic-lan");
  assert(Preferences::blobs["ba_demo_wifi/profiles2"].size()==sizeof(DemoWifiStoredProfiles));
  assert(networkGeneration==1 && bootId==physicalBoot);
  // Adding slot 1 immediately checks it without blocking LAN service or exposing its SSID.
  assert(wifiHotspotScanning && WiFi.scans==1 && WiFi.scanFilter=="synthetic-hotspot");
  assert(WiFi.scanTimeout==5000);
  const unsigned handled=api.handled;tick(100);
  assert(api.handled==handled+1 && WiFi.connected);
  WiFi.completeScan({});tick();
  assert(!wifiHotspotScanning && WiFi.starts==connectedStarts && WiFi.connected);
  tick(29999);assert(WiFi.scans==1);tick(1);
  assert(WiFi.scans==2 && wifiHotspotScanning && WiFi.connected);
  // A visible hotspot migrates a still-connected LAN without changing physical boot identity.
  WiFi.completeScan({"synthetic-hotspot"});tick();
  assert(WiFi.attempted=="synthetic-hotspot" && wifiActiveProfile==1);
  assert(clockUdp.stops==1 && discoveryUdp.stops==1);
  WiFi.connected=true;tick();
  assert(networkGeneration==2 && bootId==physicalBoot);
  assert(Preferences::numbers["ba_demo_wifi/last_profile"]==1);
  // Loading retains both credentials; last LAN success cannot override hotspot priority at boot.
  wifiProfiles[0]={};wifiProfiles[1]={};loadDemoProfiles("tof");
  assert(wifiProfiles[0].ssid=="synthetic-lan" && wifiProfiles[1].ssid=="synthetic-hotspot");
  // Failed persistence restores the in-memory profiles and leaves the live link alone.
  Preferences::rejectWrite=true;
  Serial.command("WIFI_SLOT\t1\treplacement\tsecret-new-password\n");tick();
  assert(wifiProfiles[1].ssid=="synthetic-hotspot");Preferences::rejectWrite=false;
  Serial.command("WIFI_SLOT\t1\tbad\tshort\n");tick();
  assert(wifiProfiles[1].ssid=="synthetic-hotspot");
  Serial.command("WIFI_SLOT\t1\tsynthetic-lan\tsecret-new-password\n");tick();
  assert(wifiProfiles[1].ssid=="synthetic-hotspot");
  // Hotspot loss first retries the preferred slot, then falls back after its bounded timeout.
  WiFi.connected=false;tick();tick(1000);
  assert(WiFi.attempted=="synthetic-hotspot");tick(12000);tick(1000);
  assert(WiFi.attempted=="synthetic-lan");
  WiFi.connected=true;tick();
  assert(networkGeneration==3 && wifiHotspotCheckDelayMs==30000);
  const unsigned normalScans=WiFi.scans;
  tick(29999);assert(WiFi.scans==normalScans && WiFi.connected);tick(1);
  assert(WiFi.scans==normalScans+1 && wifiHotspotScanning);
  // A visible hotspot with a failed association gets the longer migration cooldown.
  WiFi.completeScan({"synthetic-hotspot"});tick();
  assert(wifiActiveProfile==1 && wifiHotspotCheckDelayMs==120000);
  tick(12000);tick(1000);assert(wifiActiveProfile==0);
  WiFi.connected=true;tick();
  assert(networkGeneration==4 && wifiHotspotCheckDelayMs==120000);
  const unsigned cooldownScans=WiFi.scans;
  tick(119999);assert(WiFi.scans==cooldownScans && WiFi.connected);tick(1);
  assert(WiFi.scans==cooldownScans+1 && wifiHotspotScanning);
  // Transient scan failure, unrelated SSIDs and hung scans leave LAN usable.
  WiFi.scanResult=WIFI_SCAN_FAILED;tick();assert(WiFi.connected && networkWasReady);
  tick(120000);WiFi.completeScan({"unrelated-network"});tick();
  assert(WiFi.connected && wifiActiveProfile==0);
  tick(120000);const unsigned beforeStop=WiFi.scanStops;
  tick(5000);assert(WiFi.scanStops==beforeStop+1 && !wifiHotspotScanning && WiFi.connected);
  // Explicit retry cancels an in-flight scan, then attempts the preferred hotspot.
  tick(120000);assert(wifiHotspotScanning);
  Serial.command("WIFI_RETRY\n");tick();
  assert(!wifiHotspotScanning && WiFi.scanStops==beforeStop+2);tick(1000);
  assert(wifiActiveProfile==1 && WiFi.attempted=="synthetic-hotspot");
  // Both absent: bounded attempts alternate and backoff reaches its eight-second cap.
  for(unsigned i=0;i<5;++i){
    const unsigned starts=WiFi.starts;tick(12000);
    const uint32_t wait=1000u<<min(i,3u);
    assert(!wifiAttempting && wifiBackoffMs==wait);
    tick(wait-1);assert(WiFi.starts==starts);tick(1);
    assert(WiFi.starts==starts+1 && wifiAttempting);
  }
  // Millisecond wrap still reaches the attempt timeout without negative deadlines.
  fakeNow=0xfffffff0u;startDemoWifiAttempt(0);tick(12000);
  assert(!wifiAttempting && wifiBackoffMs==8000);
  assert(bootId==physicalBoot);
  checkNoCredentials();
  Preferences::numbers["ba_demo_wifi/last_profile"]=0;
  resetRuntime(false);beginDemoNetwork("tof","synthetic-firmware");
  assert(WiFi.attempted=="synthetic-hotspot" && wifiActiveProfile==1);
  // Hotspot absent at startup: timeout proceeds to LAN, even with last_profile=0.
  tick(12000);tick(999);assert(wifiActiveProfile==1);tick(1);
  assert(WiFi.attempted=="synthetic-lan" && wifiActiveProfile==0);
  WiFi.connected=true;tick();checkNoCredentials();
  assert(wifiHotspotCheckDelayMs==30000);
  // Failed async scan start is also non-disruptive.
  WiFi.scanStartResult=WIFI_SCAN_FAILED;tick(29999);assert(WiFi.scans==0);tick(1);
  assert(!wifiHotspotScanning && WiFi.connected && networkWasReady);
  // Single-slot legacy camera credentials remain usable with no preference scans.
  resetRuntime();
  Preferences::strings["tof4m_wifi/ssid"]="legacy-camera-lan";
  Preferences::strings["tof4m_wifi/password"]="secret-camera-password";
  beginDemoNetwork("camera","synthetic-firmware");
  assert(wifiProfiles[0].ssid=="legacy-camera-lan" && demoProfileCount()==1);
  assert(WiFi.attempted=="legacy-camera-lan");WiFi.connected=true;tick();tick(120000);
  assert(WiFi.scans==0 && networkWasReady);
  assert(Serial.output.find("secret-")==std::string::npos);
  std::puts("PASS: hotspot-first boot, async LAN migration, fallback, scan failure/cancel/cooldown, legacy migration, persistence, capped retry, uint32 wrap, no credentials");
}
