string drone_id
bool radio_connected
bool wifi_connected
# Battery voltage, volts. 0.0 means "no reading yet" (sim never publishes
# this, and real hardware hasn't always received a /status message yet) --
# a real battery voltage is never exactly 0, so this doubles as a validity
# flag without a separate bool.
float32 battery_voltage
