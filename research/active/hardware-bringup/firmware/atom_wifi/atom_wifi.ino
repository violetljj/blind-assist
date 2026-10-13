#include <Arduino.h>
#include <atomic>
#include "demo_wifi.h"
#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_heap_caps.h"
#include "lwip/sockets.h"
#include "lwip/tcp.h"
#include "atom_imu.h"
#include "atom_mjpeg_packet.h"
static bool cameraReady=false;
static uint32_t sequence=0;
static httpd_handle_t streamServer=nullptr;
// HTTP callbacks run on another task; release on every return/error path.
static std::atomic<unsigned> cameraStreamClients{0};
struct CameraStreamLease {
  CameraStreamLease(){cameraStreamClients.fetch_add(1);}
  ~CameraStreamLease(){cameraStreamClients.fetch_sub(1);}
};
struct CameraFrameLease {
  camera_fb_t* frame;
  explicit CameraFrameLease(camera_fb_t* value):frame(value) {}
  ~CameraFrameLease(){release();}
  void release(){if(frame){esp_camera_fb_return(frame);frame=nullptr;}}
  CameraFrameLease(const CameraFrameLease&)=delete;
  CameraFrameLease& operator=(const CameraFrameLease&)=delete;
};
static void* resizeMjpegPacket(void* buffer,size_t bytes) {
  return heap_caps_realloc(buffer,bytes,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
}
static esp_err_t stream(httpd_req_t* req) {
  if(!cameraReady){httpd_resp_set_status(req,"503 Service Unavailable");return httpd_resp_send(req,"camera_not_ready",HTTPD_RESP_USE_STRLEN);}
  CameraStreamLease lease;
  int one=1;setsockopt(httpd_req_to_sockfd(req),IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
  httpd_resp_set_type(req,"multipart/x-mixed-replace;boundary=BAFRAME");
  httpd_resp_set_hdr(req,"Cache-Control","no-store");
  AtomMjpegPacket packet(resizeMjpegPacket,heap_caps_free);
  // A send duration is known only afterward. -1 means no previous frame in this stream.
  int64_t previousSendSequence=-1,previousSendUs=-1;
  while(WiFi.status()==WL_CONNECTED) {
    int64_t getStarted=esp_timer_get_time();
    camera_fb_t* frame=esp_camera_fb_get();if(!frame)return ESP_FAIL;
    CameraFrameLease frameLease(frame);
    int64_t ready=esp_timer_get_time();
    int64_t capture=(int64_t)frame->timestamp.tv_sec*1000000+frame->timestamp.tv_usec;
    const size_t jpegLength=frame->len;
    int64_t copyStarted=esp_timer_get_time();
    if(!packet.copyJpeg(frame->buf,jpegLength)) {
      // Do not substitute a smaller/recompressed image when PSRAM is exhausted.
      Serial.println("{\"type\":\"error\",\"message\":\"mjpeg_packet_allocation_or_size\"}");
      return ESP_ERR_NO_MEM;
    }
    int64_t copyUs=esp_timer_get_time()-copyStarted;
    frameLease.release(); // Network send now reads only the immutable owned copy.
    const uint32_t frameSequence=++sequence;
    int64_t sendStarted=esp_timer_get_time();
    char header[AtomMjpegPacket::headerCapacity];int length=snprintf(header,sizeof(header),
      "\r\n--BAFRAME\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\nX-Sequence-Id: %s\r\nX-Frame-Sequence: %lu\r\nX-Capture-Timestamp-Us: %lld\r\nX-Jpeg-Ready-Timestamp-Us: %lld\r\nX-Device-Send-Start-Timestamp-Us: %lld\r\nX-Frame-Get-Wait-Us: %lld\r\nX-Jpeg-Copy-Us: %lld\r\nX-Wifi-Rssi-Dbm: %d\r\nX-Previous-Response-Write-Valid: %s\r\nX-Previous-Frame-Sequence: %lld\r\nX-Previous-Response-Write-Duration-Us: %lld\r\nX-Tof-Valid: false\r\n\r\n",
      (unsigned)jpegLength,bootId.c_str(),(unsigned long)frameSequence,(long long)capture,(long long)ready,
      (long long)sendStarted,(long long)(ready-getStarted),(long long)copyUs,
      (int)WiFi.RSSI(),previousSendSequence>=0?"true":"false",
      (long long)previousSendSequence,(long long)previousSendUs);
    if(length<=0 || length>=(int)sizeof(header) || !packet.prependHeader(header,length))return ESP_FAIL;
    // A single HTTP chunk avoids separate small header/body chunks with TCP_NODELAY.
    esp_err_t result=httpd_resp_send_chunk(req,packet.data(),packet.size());
    previousSendSequence=frameSequence;previousSendUs=esp_timer_get_time()-sendStarted;
    if(result!=ESP_OK)return result;
  }
  return httpd_resp_send_chunk(req,nullptr,0);
}

static bool configureCameraTiming() {
  sensor_t* sensor=esp_camera_sensor_get();
  if(!sensor || sensor->id.PID!=OV3660_PID){
    Serial.println("{\"type\":\"error\",\"message\":\"camera_timing_sensor_mismatch\"}");
    return false;
  }
  // OV3660 0x471D[1:0]=2 emits VSYNC at field start, independent of JPEG EOF.
  // The camera driver still timestamps DMA acquisition start, not exposure.
  const int result=sensor->set_reg(sensor,0x471d,0x03,0x02);
  const int actual=sensor->get_reg(sensor,0x471d,0x03);
  Serial.printf("{\"type\":\"camera_timing\",\"sensor_pid\":%u,\"vsync_mode\":%d,\"status\":%d}\n",(unsigned)sensor->id.PID,actual,result);
  return result==0 && actual==2;
}

void setup() {
  Serial.begin(115200);delay(1500);
  beginDemoNetwork("camera","atom-camera-wifi-v1");
  if(!psramFound()){Serial.println("{\"type\":\"error\",\"message\":\"psram_missing\"}");return;}
  pinMode(18,OUTPUT);digitalWrite(18,LOW);delay(500);
  camera_config_t c={};
  c.pin_pwdn=-1;c.pin_reset=-1;c.pin_xclk=21;c.pin_sccb_sda=12;c.pin_sccb_scl=9;
  c.pin_d7=13;c.pin_d6=11;c.pin_d5=17;c.pin_d4=4;c.pin_d3=48;c.pin_d2=46;c.pin_d1=42;c.pin_d0=3;
  c.pin_vsync=10;c.pin_href=14;c.pin_pclk=40;c.xclk_freq_hz=20000000;
  c.ledc_timer=LEDC_TIMER_0;c.ledc_channel=LEDC_CHANNEL_0;
  c.pixel_format=PIXFORMAT_JPEG;c.frame_size=FRAMESIZE_VGA;c.jpeg_quality=12;c.fb_count=2;
  c.fb_location=CAMERA_FB_IN_PSRAM;c.grab_mode=CAMERA_GRAB_LATEST;c.sccb_i2c_port=AtomImu::cameraSccbPort;
  esp_err_t result=esp_camera_init(&c);cameraReady=result==ESP_OK;
  Serial.printf("{\"type\":\"camera_init\",\"status\":%d}\n",(int)result);
  if(cameraReady)cameraReady=configureCameraTiming();
  // Camera bus allocation and sensor configuration precede BMI270 samples.
  AtomImu::begin();

  api.on("/api/camera-timing",HTTP_GET,[]{
    sensor_t* sensor=esp_camera_sensor_get();
    if(!sensor){api.send(503,"text/plain","no_sensor");return;}
    const int mode=sensor->id.PID==OV3660_PID?sensor->get_reg(sensor,0x471d,0x03):-1;
    const int quality=sensor->id.PID==OV3660_PID?sensor->get_reg(sensor,0x4407,0x3f):-1;
    const int width=sensor->id.PID==OV3660_PID?sensor->get_reg(sensor,0x4602,0xffff):-1;
    char body[320];snprintf(body,sizeof(body),"{\"boot_id\":\"%s\",\"camera_ready\":%s,\"sensor_pid\":%u,\"vsync_mode\":%d,\"jpeg_quality\":%d,\"jpeg_line_width\":%d,\"timestamp_semantics\":\"driver_frame_start;exposure_unverified\"}",bootId.c_str(),cameraReady?"true":"false",(unsigned)sensor->id.PID,mode,quality,width);
    api.send(cameraReady && mode==2 && quality>=0 && width>=0?200:503,"application/json",body);
  });

  httpd_config_t config=HTTPD_DEFAULT_CONFIG();config.server_port=81;config.ctrl_port=32769;
  config.send_wait_timeout=1;config.recv_wait_timeout=1;config.max_open_sockets=2;config.lru_purge_enable=true;
  if(httpd_start(&streamServer,&config)==ESP_OK){
    httpd_uri_t uri={};uri.uri="/stream";uri.method=HTTP_GET;uri.handler=stream;httpd_register_uri_handler(streamServer,&uri);
  }
}
void loop(){
  demoNetworkLoop(true,cameraStreamClients.load()>0 || AtomImu::hasTransportClients());
  AtomImu::loop();delay(1);
}
