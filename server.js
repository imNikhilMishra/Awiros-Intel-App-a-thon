const express = require('express');
const cors = require('cors');
const multer = require('multer'); // To handle file uploads
const app = express();
const { spawn } = require('child_process');
const path = require('path');

app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 5000;

const storage = multer.memoryStorage();
const upload = multer({ storage });


app.post('/', upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded' });
    }
    const userImageBuffer = req.file.buffer;

    const userImageBase64 = userImageBuffer.toString('base64');

    const datasetImageFileName = 'Virat-Kohli-Cover-Drive.jpeg'; // Specify the filename

    // Create the absolute path by joining __dirname with the relative path to the image file
    const datasetImageFilePath = path.join(__dirname, 'Images', datasetImageFileName);
    

    const data = {
        userImage: userImageBase64,
        datasetImage: datasetImageFilePath,
    };
    
    const dataJson = JSON.stringify(data);
    const pythonProcess = spawn('python', ['./Model/ssi-model.py', JSON.stringify(dataJson)]);
    // pythonProcess.stdin.write(JSON.stringify(data));
    // pythonProcess.stdin.end();

    let mlResponse = '';

    pythonProcess.stdout.on('data', (data) => {
        mlResponse += data;
    });
    
    pythonProcess.on('close', () => {
        // Process the mlResponse and send it to the frontend
        res.status(200).json({ mlResponse });
    });
});

app.listen(PORT, () => {
    console.log(`Server running on port: ${PORT}`);
});
