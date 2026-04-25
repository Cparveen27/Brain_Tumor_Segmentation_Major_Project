-- Create a new database
CREATE DATABASE LifeCareDB;

-- Switch to the new database
USE LifeCareDB;

-- Create the users table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    mobile VARCHAR(15) NOT NULL,
    gender ENUM('M', 'F', 'O') NOT NULL,
    age INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create a table to store prediction results including user email, input image, mask, overlay image, class name, and date
CREATE TABLE prediction_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_email VARCHAR(100) NOT NULL,
    input_image_filename VARCHAR(255) NOT NULL,
    mask_filename VARCHAR(255),
    overlay_filename VARCHAR(255),
    class_name VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);