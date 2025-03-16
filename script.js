document.getElementById("prediction-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    // Collect form data
    let formData = {
        MedInc: parseFloat(document.getElementById("MedInc").value),
        HouseAge: parseFloat(document.getElementById("HouseAge").value),
        AveRooms: parseFloat(document.getElementById("AveRooms").value),
        AveBedrms: parseFloat(document.getElementById("AveBedrms").value),
        Population: parseInt(document.getElementById("Population").value),
        AveOccup: parseFloat(document.getElementById("AveOccup").value),
    };

    // Show loading text
    document.getElementById("result").innerHTML = "Predicting...";

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData),
        });

        const data = await response.json();
        document.getElementById("result").innerHTML = `Predicted Price: $${data["Predicted House Price"]}`;
    } catch (error) {
        console.error("Error fetching prediction:", error);
        document.getElementById("result").innerHTML = "Error fetching prediction";
    }
});
