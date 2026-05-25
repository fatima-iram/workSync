const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");

const app = express();

app.use(cors());
app.use(bodyParser.json());

app.post("/save-email", (req, res) => {
    const { email } = req.body;

    console.log("Email received:", email);

    res.json({
        success: true,
        message: "Email saved successfully"
    });
});

app.listen(5000, () => {
    console.log("Server running on port 5000");
});