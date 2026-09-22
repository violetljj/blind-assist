#include <Arduino.h>
#include "demo_wifi.h"
#include "esp_camera.h"
#include "esp_http_server.h"
#include "lwip/sockets.h"
#include "lwip/tcp.h"
static bool cameraReady=false;
static uint32_t sequence=0;
static httpd_handle_t streamServer=nullptr;
static esp_err_t stream(httpd_req_t* req) {
  if(!cameraReady){httpd_resp_set_status(req,"503 Service Unavailable");return httpd_resp_send(req,"camera_not_ready",HTTPD_RESP_USE_STRLEN);}
  int one=1;setsockopt(httpd_req_to_sockfd(req),IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
  httpd_resp_set_type(req,"multipart/x-mixed-replace;boundary=BAFRAME");
  httpd_resp_set_hdr(req,"Cache-Control","no-store");
  while(WiFi.status()==WL_CONNECTED) {
    camera_fb_t* frame=esp_camera_fb_get();if(!frame)return ESP_FAIL;
    int64_t ready=esp_timer_get_time();
    int64_t capture=(int64_t)frame->timestamp.tv_sec*1000000+frame->timestamp.tv_usec;
    char header[512];int length=snprintf(header,sizeof(header),
      "\r\n--BAFRAME\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\nX-Sequence-Id: %s\r\nX-Frame-Sequence: %lu\r\nX-Capture-Timestamp-Us: %lld\r\nX-Jpeg-Ready-Timestamp-Us: %lld\r\nX-Device-Send-Start-Timestamp-Us: %lld\r\nX-Tof-Valid: false\r\n\r\n",
      (unsigned)frame->len,bootId.c_str(),(unsigned long)++sequence,(long long)capture,(long long)ready,(long long)esp_timer_get_time());
    esp_err_t result=length>0 && length<(int)sizeof(header)?httpd_resp_send_chunk(req,header,length):ESP_FAIL;
    if(result==ESP_OK)result=httpd_resp_send_chunk(req,(const char*)frame->buf,frame->len);
    esp_camera_fb_return(frame);
    if(result!=ESP_OK)return result;
  }
  return httpd_resp_send_chunk(req,nullptr,0);
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
  c.fb_location=CAMERA_FB_IN_PSRAM;c.grab_mode=CAMERA_GRAB_LATEST;c.sccb_i2c_port=0;
  esp_err_t result=esp_camera_init(&c);cameraReady=result==ESP_OK;
  Serial.printf("{\"type\":\"camera_init\",\"status\":%d}\n",(int)result);
  httpd_config_t config=HTTPD_DEFAULT_CONFIG();config.server_port=81;config.ctrl_port=32769;
  config.send_wait_timeout=1;config.recv_wait_timeout=1;config.max_open_sockets=2;config.lru_purge_enable=true;
  if(httpd_start(&streamServer,&config)==ESP_OK){
    httpd_uri_t uri={};uri.uri="/stream";uri.method=HTTP_GET;uri.handler=stream;httpd_register_uri_handler(streamServer,&uri);
  }
}
void loop(){demoNetworkLoop();delay(1);}
