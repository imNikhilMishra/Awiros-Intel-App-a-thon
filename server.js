const express = require('express');
const cors = require('cors');
const multer = require('multer'); // To handle file uploads
const app = express();
const { spawn } = require('child_process');

app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 5000;

const storage = multer.memoryStorage();
const upload = multer({ storage });

app.post('/', upload.single('image'), (req, res) => {
    const imageBuffer = req.file.buffer;

    // Send the image directly to your ML model
    sendImageToMLModel(imageBuffer, (mlResponse) => {
        res.status(200).json({ message: 'Image sent to the ML model', mlResponse });
    });
});

const sendImageToMLModel=(imageData, callback) =>{
    // Replace this with code to send the image data to your ML model
    // You might use an HTTP request or any other appropriate method
    // For example, you can use the 'spawn' method to execute a Python script that communicates with your model.
    // Make sure your ML model API endpoint and data format are appropriately configured.

    const pythonProcess = spawn('python', ['./Model/ssim_posturecorrection.py'], {
        input: imageData
    });

    let mlResponse = '';

    pythonProcess.stdout.on('data', (data) => {
        mlResponse += data;
    });

    pythonProcess.on('close', () => {
        callback(mlResponse);
    });
}

app.listen(PORT, () => {
    console.log(`Server running on port: ${PORT}`);
});
