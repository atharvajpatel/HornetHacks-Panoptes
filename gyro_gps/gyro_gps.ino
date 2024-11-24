/* Get tilt angles on X and Y, and rotation angle on Z
  Angles are given in degrees
License: MIT
*/
#include "Wire.h"
#include "Arduino.h"
#include <MPU6050_light.h>
#include "HT_TinyGPS++.h"
#include "HT_st7735.h"

// CONSTS
#define MPU_DELAY_TIME 10 // in ms
#define GPS_DELAY_TIME 5000 // in ms
#define VGNSS_CTRL 3
#define DEBUG_MODE false

// OBJECTS
MPU6050 mpu(Wire);
TinyGPSPlus GPS;
HT_st7735 st7735;

// GLOBALS
unsigned long timer1 = 0;
unsigned long timer2 = 0;
const float deg_to_rad = PI / 180;
float altitude_sens = 0; // Set to a constant since no altitude sensor

void GPS_send(void)
{
  if(Serial1.available()>0)
  {
    if(Serial1.peek()!='\n')
    {
      GPS.encode(Serial1.read());
    }
    else
    {
      Serial1.read();
      if(GPS.time.second()==0)
      {
        return;
      }
#if DEBUG_MODE
      Serial.printf(" %02d:%02d:%02d.%02d",GPS.time.hour(),GPS.time.minute(),GPS.time.second(),GPS.time.centisecond());
      Serial.print("LAT: ");
      Serial.print(GPS.location.lat(), 6);
      Serial.print(", LON: ");
      Serial.print(GPS.location.lng(), 6);
      Serial.println();
#endif
      st7735.st7735_fill_screen(ST7735_BLACK);
      st7735.st7735_write_str(0, 0, (String)"GPS_test");
      String time_str = (String)GPS.time.hour() + ":" + (String)GPS.time.minute() + ":" + (String)GPS.time.second()+ ":"+(String)GPS.time.centisecond();
      st7735.st7735_write_str(0, 20, time_str);
      String latitude = "LAT: " + (String)GPS.location.lat();
      st7735.st7735_write_str(0, 40, latitude);
      String longitude  = "LON: "+  (String)GPS.location.lng();
      st7735.st7735_write_str(0, 60, longitude);
      
      while(Serial1.read()>0);
    }
  }
}

void setup() {
  Serial.begin(9600);
  Wire.begin();
  byte status = mpu.begin();
#if DEBUG_MODE
  Serial.print(F("MPU6050 status: "));
  Serial.println(status);
#endif
  while (status != 0) { } // stop everything if could not connect to MPU6050
#if DEBUG_MODE
  Serial.println(F("Calculating offsets, do not move MPU6050"));
#endif
  delay(1000);
  mpu.calcOffsets(); // gyro and accelero

  // GPS setup
  pinMode(VGNSS_CTRL,OUTPUT);
  digitalWrite(VGNSS_CTRL,HIGH);
  Serial1.begin(115200,SERIAL_8N1,33,34);

  // TFT Setup
  st7735.st7735_init();
  delay(100);
  st7735.st7735_fill_screen(ST7735_BLACK);
  delay(100);
  st7735.st7735_write_str(0, 0, (String)"GPS_test");
  
#if DEBUG_MODE
  Serial.println("GPS_test");
#endif
}
void loop() {
  mpu.update();
  unsigned long current_time = millis();
  if ((current_time - timer1) > MPU_DELAY_TIME) { // print data every 10ms
#if DEBUG_MODE
    Serial.print("X : ");
    Serial.print(mpu.getAngleX());
    Serial.print("tY : ");
    Serial.print(mpu.getAngleY());
    Serial.print("tZ : ");
    Serial.println(mpu.getAngleZ());
#else
    Serial.print(GPS.location.lat(), 6);
    Serial.print(",");
    Serial.print(GPS.location.lng(), 6);
    Serial.print(",");
    Serial.print(altitude_sens, 6);
    Serial.print(",");
    Serial.print(mpu.getAngleX()* deg_to_rad, 6);
    Serial.print(",");
    Serial.print(mpu.getAngleY()* deg_to_rad, 6);
    Serial.print(",");
    Serial.println(mpu.getAngleZ()* deg_to_rad, 6);
#endif
    timer1 = millis();
  }
  if ((current_time - timer2) > GPS_DELAY_TIME) { // print data every 5s
    GPS_send();
    timer2 = millis();
  }
}
