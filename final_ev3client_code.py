#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, GyroSensor
from pybricks.parameters import Port, Stop
from pybricks.robotics import DriveBase
from pybricks.tools import wait
import socket
import gc
import time

# Initialize the EV3 Brick
ev3 = EV3Brick()

# Initialize the gyroscope sensor
gyro = GyroSensor(Port.S2)  # Ensure the gyroscope is connected to the correct port

# Reset the gyroscope to 0 degrees
gyro.reset_angle(0)

# Initialize the drive base
left_motor = Motor(Port.C)
right_motor = Motor(Port.B)
drop_off = Motor(Port.A)
pick_up = Motor(Port.D)

robot = DriveBase(left_motor, right_motor, wheel_diameter=56, axle_track=114)
pick_up.run(400)

# Set the server's IP address and port number
HOST, PORT = '192.168.252.28', 5000

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
bufSize = 1024

# Functions
def receive_instructions():
    try:
        received_command = sock.recv(bufSize)  # The buffer size is 1024 bytes
        if received_command:
            print("Received command: {}".format(received_command))
        return received_command
    except Exception as e:
        print("No message or error occurred: ", e)
        return None

def turn_mode(turn_angle):
    # Reset the gyroscope to 0 degrees before starting the turn
    gyro.reset_angle(0)
    
    # Determine the turn speed
    turn_speed = 300  # Adjust as needed

    # Start turning the robot
    if turn_angle > 0:
        left_motor.run(turn_speed)
        right_motor.run(-turn_speed)
    else:
        left_motor.run(-turn_speed)
        right_motor.run(turn_speed)
    
    # Continue turning until the target angle is reached
    while True:
        current_angle = gyro.angle()
        
        # Check if the target angle is reached maybe minus 2
        if abs(current_angle) >= abs(turn_angle)-2:
            left_motor.stop(Stop.BRAKE)
            right_motor.stop(Stop.BRAKE)
            ev3.speaker.beep()  # Beep to indicate the robot has reached the target angle
            print("Beep! Registered a {} degree turn.".format(current_angle))
            break
        

def execute_instruction(instruction):
    # Instruction format: "2,turn_angle,distance"
    parts = instruction.split(',')
    if parts[0] == '2':
        turn_angle = float(parts[1])
        distance = float(parts[2])
        print("Received instruction to turn {} degrees and move {} mm".format(turn_angle, distance))
        ev3.speaker.beep()  # Beep to indicate instruction receipt

        # Use turn_mode to perform the turn using the gyroscope
        turn_mode(turn_angle)

        # Move straight for the specified distance
        robot.straight((distance * 1.8 * -1) + 30)
        
        # Ensure the robot completes the straight movement before proceeding
        robot.stop()

        # Send acknowledgment to the server 
        sock.sendall("ACK".encode())
        print("Sent ACK")  # Debugging: Confirm acknowledgment sent
        wait(1000)  # Add a small delay to ensure the ACK is sent
    
    elif parts[0] == '1':
        print("We are in deposit now")
        turn_angle = float(parts[1])
        distance = float(parts[2])
        print("Received instruction to turn {} degrees and move {} mm".format(turn_angle, distance))
        ev3.speaker.beep()  # Beep to indicate instruction receipt

        # Use turn_mode to perform the turn using the gyroscope
        turn_mode(turn_angle)

        # Move straight for the specified distance
        robot.straight((distance * 1.75 * -1) + 30)
        
        # Ensure the robot completes the straight movement before proceeding
        robot.stop()

        # Move straight for the specified distance
        robot.straight((100 * 1.75 ))

        # Turn the motor by -30 degrees
        drop_off.run_target(speed=200, target_angle=-40)
        # Wait a bit to ensure the motor completes the action
        wait(1000)
        drop_off.run_target(speed=200, target_angle=0)
        # Wait a bit to ensure the motor completes the action
        wait(1000)
        drop_off.run_target(speed=200, target_angle=-40)
        # Wait a bit to ensure the motor completes the action
        wait(1000)
        drop_off.run_target(speed=200, target_angle=0)

        # Send acknowledgment to the server 
        sock.sendall("ACK".encode())
        print("Sent ACK")  # Debugging: Confirm acknowledgment sent
        wait(1000)  # Add a small delay to ensure the ACK is sent

# Main
try:
    sock.connect((HOST, PORT))
    print("Connected to server.")

    # Send the initial position message to the server
    start_message = '0,0'  # Example starting position
    sock.sendall(start_message.encode())
    time.sleep(1)  # Add a small delay to ensure the initial message is sent

    while True:
        received_command = receive_instructions()
        while received_command:
            instruction = received_command.decode()
            execute_instruction(instruction)
            received_command = receive_instructions()
        gc.collect()
except Exception as e:
    print("Failed to connect or error occurred: ", e)

finally:
    sock.close()
    print("Socket closed. Program finished.")
