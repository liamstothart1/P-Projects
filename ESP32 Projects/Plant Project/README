Plant Monitoring System with OLED Display and Deep Sleep

This project implements a Plant Monitoring System using an OLED display, LDR sensor, temperature sensor, and soil moisture sensor on an ESP32 board. The system displays environmental conditions and alerts on the OLED screen, and utilizes deep sleep mode to conserve power between measurements.

Features

	•	Light Monitoring: Measures ambient light using an LDR sensor and provides warnings if the light is too high or too low.
	•	Temperature Monitoring: Measures ambient temperature and provides warnings if the temperature is too high or too low.
	•	Soil Moisture Monitoring: Measures soil moisture and provides warnings if the soil is too wet or too dry.
	•	OLED Display: Displays warnings or status messages, and shows current sensor readings.
	•	Deep Sleep Mode: Utilizes ESP32 deep sleep mode to conserve power between display updates.

Hardware Requirements

	•	ESP32 board
	•	SSD1306 OLED display
	•	LDR sensor
	•	Temperature sensor (e.g., LM35)
	•	Soil moisture sensor
	•	Wires and breadboard

Software Requirements

	•	Arduino IDE with ESP32 support
	•	Adafruit GFX library
	•	Adafruit SSD1306 library
	•	ESP32 library

Circuit Diagram

Connect the components as follows:

	•	LDR sensor: Connect to analog pin A0 (pin 34 in the code).
	•	Temperature sensor: Connect to analog pin A1 (pin 35 in the code).
	•	Soil moisture sensor: Connect to analog pin A2 (pin 32 in the code).
	•	SSD1306 OLED display: Connect to the I2C pins (SDA and SCL) on the ESP32.

Installation

	1.	Install Arduino IDE: Download and install the Arduino IDE from the official Arduino website.
	2.	Install ESP32 Board Support: Follow the instructions to install ESP32 board support in the Arduino IDE.
	3.	Install Libraries: Open the Arduino IDE, go to Sketch > Include Library > Manage Libraries. Install the following libraries:
	•	Adafruit GFX
	•	Adafruit SSD1306
	•	ESP32 library
	4.	Upload the Code: Connect your ESP32 board to your computer, open the provided code in the Arduino IDE, and upload it to your ESP32.

## Setup 
void setup() {
  Serial.begin(115200);
  pinMode(ldrPin, INPUT);
  pinMode(tempPin, INPUT);
  pinMode(soilMoisturePin, INPUT);

  // Initialize OLED display with I2C address
  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
    for (;;);
  }

  display.display();
  delay(2000);
  display.clearDisplay();

  // Show initial message "Hi Becks :)" for 5 seconds
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 16);
  display.println("Hi Becks:)");
  display.display();
  delay(5000);

  // Record the start time of the display
  displayStartTime = millis();
}

## Main Loop ]
void loop() {
  unsigned long currentTime = millis();
  unsigned long elapsedTime = currentTime - displayStartTime;

  int ldrValue = analogRead(ldrPin);
  int tempValue = analogRead(tempPin);
  int soilMoistureValue = analogRead(soilMoisturePin);

  float voltage = tempValue * (3.3 / 4095.0);
  float temperatureC = voltage * 100.0;
  float ldrPercentage = ((float)ldrValue / ldrTargetValue) * 100;
  float soilMoisturePercentage = ((float)soilMoistureValue / soilMoistureTargetValue) * 100;

  bool warning = false;
  int messageCount = 0;
  const char* currentWarnings[9]; // Maximum of 9 warnings (3 per type)
  int warningValues[9];
  const char* warningUnits[9];
  const char* warningDirections[9];

  // Check sensor values and add warnings if thresholds are exceeded
  if (ldrValue > ldrThresholdHigh) {
    int messageIndex = random(0, 3);
    currentWarnings[messageCount] = lightHighWarnings[messageIndex];
    warningValues[messageCount] = ldrPercentage - ((float)ldrThresholdHigh / ldrTargetValue) * 100;
    warningUnits[messageCount] = "%";
    warningDirections[messageCount] = "over";
    messageCount++;
    warning = true;
  } else if (ldrValue < ldrThresholdLow) {
    int messageIndex = random(0, 3);
    currentWarnings[messageCount] = lightLowWarnings[messageIndex];
    warningValues[messageCount] = ((float)ldrThresholdLow / ldrTargetValue) * 100 - ldrPercentage;
    warningUnits[messageCount] = "%";
    warningDirections[messageCount] = "under";
    messageCount++;
    warning = true;
  }

  // Check temperature and add warnings if thresholds are exceeded
  if (temperatureC > tempThresholdHigh) {
    int messageIndex = random(0, 3);
    currentWarnings[messageCount] = tempHighWarnings[messageIndex];
    warningValues[messageCount] = temperatureC - tempThresholdHigh;
    warningUnits[messageCount] = "C";
    warningDirections[messageCount] = "over";
    messageCount++;
    warning = true;
  } else if (temperatureC < tempThresholdLow) {
    int messageIndex = random(0, 3);
    currentWarnings[messageCount] = tempLowWarnings[messageIndex];
    warningValues[messageCount] = tempThresholdLow - temperatureC;
    warningUnits[messageCount] = "C";
    warningDirections[messageCount] = "under";
    messageCount++;
    warning = true;
  }

  // Check soil moisture and add warnings if thresholds are exceeded
  if (soilMoistureValue > soilMoistureThresholdHigh) {
    int messageIndex = random(0, 3);
    currentWarnings[messageCount] = soilMoistureHighWarnings[messageIndex];
    warningValues[messageCount] = soilMoisturePercentage - ((float)soilMoistureThresholdHigh / soilMoistureTargetValue) * 100;
    warningUnits[messageCount] = "%";
    warningDirections[messageCount] = "over";
    messageCount++;
    warning = true;
  } else if (soilMoistureValue < soilMoistureThresholdLow) {
    int messageIndex = random(0, 3);
    currentWarnings[messageCount] = soilMoistureLowWarnings[messageIndex];
    warningValues[messageCount] = ((float)soilMoistureThresholdLow / soilMoistureTargetValue) * 100 - soilMoisturePercentage;
    warningUnits[messageCount] = "%";
    warningDirections[messageCount] = "under";
    messageCount++;
    warning = true;
  }

  // Display warnings or "Plant is all good!" message
  if (warning) {
    displayWarnings(currentWarnings, messageCount, warningValues, warningUnits, warningDirections);
  } else {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(2);
    display.setTextColor(SSD1306_WHITE);
    display.println("Plant is");
    display.println("all good!");
    display.display();
    delay(5000);
  }

  // Display sensor readings for a duration, then clear display
  displayStartTime = millis(); // Reset display start time for measurement display
  while ((millis() - displayStartTime) < displayDuration) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.print("LDR: ");
    display.print(ldrPercentage, 1);
    display.println("%");
    display.setCursor(0, 16);
    display.print("Temp (C): ");
    display.println(temperatureC);
    display.setCursor(0, 32);
    display.print("Soil Moisture: ");
    display.print(soilMoisturePercentage, 1);
    display.println("%");
    display.display();
    delay(1000);
  }

  // Clear display and enter deep sleep mode to conserve power
  display.clearDisplay();
  display.display();
  Serial.println("Entering deep sleep mode");
  esp_deep_sleep_start();
}

This code continuously monitors the plant’s environment, displays warnings on the OLED screen if conditions are unfavorable, and enters deep sleep mode to conserve power between measurements. Adjust thresholds and messages as needed for your specific application.
