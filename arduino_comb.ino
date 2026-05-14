#include <Wire.h>

// ================== PIN DEFINITIONS ==================
const int PWM_R = 6;   
const int DIR_R = 5;   
const int ENCA_R = 18; 

const int PWM_L = 10;  
const int DIR_L = 9;   
const int ENCA_L = 3;  

// ================== ENCODERS ==================
volatile long countR = 0;
volatile long countL = 0;

// ================== IMU ==================
const int MPU_addr = 0x68; 
int16_t AcX, AcY, AcZ, GyX, GyY, GyZ;

long offGyZ = 0;

// ================== VARIABLES ==================
long prevCountL = 0;
long prevCountR = 0;
unsigned long prevTime = 0;

// ================== LOCKED CALIBRATION ==================
// We found this is the perfect number for your carpet!
float k = 4.92; 

float omegaIMU_filtered = 0;

// ================== SETUP ==================
void setup() {
  Serial.begin(115200);

  Wire.begin();
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);

  // ====== GYRO ZERO-CALIBRATION ======
  // Keeps the gyro steady when standing perfectly still.
  // KEEP THE ROBOT STILL FOR 1 SECOND WHEN TURNING IT ON!
  long sumGyZ = 0;
  int samples = 110;

  for (int i = 0; i < samples; i++) {
    readRawIMU();
    sumGyZ += GyZ;
    delay(5);
  }

  offGyZ = sumGyZ / samples;

  pinMode(PWM_R, OUTPUT); pinMode(DIR_R, OUTPUT);
  pinMode(PWM_L, OUTPUT); pinMode(DIR_L, OUTPUT);

  attachInterrupt(digitalPinToInterrupt(ENCA_R), []{ countR++; }, RISING);
  attachInterrupt(digitalPinToInterrupt(ENCA_L), []{ countL++; }, RISING);

  prevTime = millis();
}

// ================== LOOP ==================
void loop() {

  readRawIMU();

  unsigned long currentTime = millis();
  float dt = (currentTime - prevTime) / 1000.0;
  if (dt <= 0.001) return;

  // ====== ENCODER VELOCITIES ======
  float velL = (countL - prevCountL) / dt;
  float velR = (countR - prevCountR) / dt;

  prevCountL = countL;
  prevCountR = countR;
  prevTime = currentTime;

  // ====== EXPECTED YAW ======
  float omega_enc = (velR - velL);

  // ====== IMU YAW ======
  float omega_imu = -(GyZ - offGyZ) / 131.0;

  // ====== SCALE + FILTER ======
  // Multiply by our locked-in carpet value
  float omega_imu_scaled = omega_imu * k;
  
  // Smooth out any tiny vibrations
  omegaIMU_filtered = 0.6 * omegaIMU_filtered + 0.4 * omega_imu_scaled;

  // ====== SLIP SIGNAL ======
  float yaw_error = omega_enc - omegaIMU_filtered;
  float norm_error = yaw_error / (abs(omega_enc) + 1);

  // ====== FORWARD MOTION ======
  // Add your driving logic here
  digitalWrite(DIR_R, LOW);
  digitalWrite(DIR_L, HIGH);
  analogWrite(PWM_L, 200);
  analogWrite(PWM_R, 200);

  // ================== COMBINED SERIAL OUTPUT ==================
  // Format matches exactly what the Python script expects:
  // encL, encR, AcX, AcY, AcZ, omega_enc, omega_imu_filtered, slip_spike, k_scaled
  Serial.print(countL); Serial.print(",");
  Serial.print(countR); Serial.print(",");
  Serial.print(AcX); Serial.print(",");
  Serial.print(AcY); Serial.print(",");
  Serial.print(AcZ); Serial.print(",");
  Serial.print(omega_enc); Serial.print(",");
  Serial.print(omegaIMU_filtered); Serial.print(",");
  Serial.print(norm_error * 500); Serial.print(",");
  Serial.println(k * 100);

  delay(20);
}

// ================== IMU READ ==================
void readRawIMU() {
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_addr, 14, true);

  AcX = Wire.read() << 8 | Wire.read();
  AcY = Wire.read() << 8 | Wire.read();
  AcZ = Wire.read() << 8 | Wire.read();

  Wire.read(); Wire.read(); // Skip Temperature

  GyX = Wire.read() << 8 | Wire.read();
  GyY = Wire.read() << 8 | Wire.read();
  GyZ = Wire.read() << 8 | Wire.read();
}