import React, { useEffect, useState } from "react";
import './App.css';
function App() {
  const [districts, setDistricts] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState("");
  const [wards, setWards] = useState([]);
  const [selectedWard, setSelectedWard] = useState("");
  const [squares, setSquares] = useState("");
  const [bedrooms, setBedrooms] = useState("");
  const [bathrooms, setBathrooms] = useState("");
  const [direction, setDirection] = useState("");
  const [balcony, setBalcony] = useState("");
  const [predictedPrice, setPredictedPrice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [investors, setInvestors] = useState([]);
  const [selectedInvestor, setSelectedInvestor] = useState("");

  useEffect(() => {
    fetch("http://localhost:5000/api/districts")
      .then((res) => res.json())
      .then((data) => setDistricts(data))
      .catch((err) => console.error(err));
  }, []);

  useEffect(() => {
    if (selectedDistrict) {
      fetch(`http://localhost:5000/api/wards/${selectedDistrict}`)
        .then((res) => res.json())
        .then((data) => setWards(data))
        .catch((err) => console.error(err));
    } else {
      setWards([]);
    }
  }, [selectedDistrict]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    const data = {
      squares,
      bedrooms,
      bathrooms,
      direction,
      balcony,
      district: selectedDistrict,
      ward: selectedWard,
      investor: selectedInvestor

    };

    try {
      const response = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        const result = await response.json();
        setPredictedPrice(result.predicted_price);
      } else {
        console.error("Error fetching prediction:", response.statusText);
      }
    } catch (error) {
      console.error("Error fetching prediction:", error);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
  fetch("http://localhost:5000/api/investors")
    .then((res) => res.json())
    .then((data) => setInvestors(data))
    .catch((err) => console.error(err));
  }, []);

  return (
    <div className="container">
      <h1>Dự đoán giá bất động sản</h1>
      <form onSubmit={handleSubmit}>
        <div className="dropdowns">
          <div>
            <label>Quận / Huyện:</label>
            <select
              value={selectedDistrict}
              onChange={(e) => setSelectedDistrict(e.target.value)}
            >
              <option value="">-- Chọn quận --</option>
              {districts.map((d, index) => (
                <option key={index} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label>Phường / Xã:</label>
            <select
              value={selectedWard}
              onChange={(e) => setSelectedWard(e.target.value)}
              disabled={!wards.length}
            >
              <option value="">-- Chọn phường --</option>
              {wards.map((w, index) => (
                <option key={index} value={w}>
                  {w}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div>
        <label>Chủ đầu tư:</label>
          <select
            value={selectedInvestor}
            onChange={(e) => setSelectedInvestor(e.target.value)}
            required
          >
            <option value="">-- Chọn chủ đầu tư --</option>
            {investors.map((inv, index) => (
              <option key={index} value={inv}>
                {inv}
              </option>
            ))}
          </select>
        </div>

        <div className="input-fields">
          <div>
            <label>Diện tích (m²):</label>
            <input
              type="number"
              value={squares}
              onChange={(e) => setSquares(e.target.value)}
              required
            />
          </div>

          <div>
            <label>Số phòng ngủ:</label>
            <input
              type="number"
              value={bedrooms}
              onChange={(e) => setBedrooms(e.target.value)}
              required
            />
          </div>

          <div>
            <label>Số phòng tắm:</label>
            <input
              type="number"
              value={bathrooms}
              onChange={(e) => setBathrooms(e.target.value)}
              required
            />
          </div>

          <div>
            <label>Hướng:</label>
            <select
              value={direction}
              onChange={(e) => setDirection(e.target.value)}
              required
            >
              <option value="">-- Chọn hướng --</option>
              <option value="Tây-Bắc">Tây-Bắc</option>
              <option value="Đông-Nam">Đông-Nam</option>
              <option value="Tây-Nam">Tây-Nam</option>
              <option value="Đông-Bắc">Đông-Bắc</option>
              <option value="Bắc">Bắc</option>
              <option value="Nam">Nam</option>
              <option value="Đông">Đông</option>
              <option value="Tây">Tây</option>
            </select>
          </div>

          <div>
            <label>Ban công:</label>
            <select
              value={balcony}
              onChange={(e) => setBalcony(e.target.value)}
              required
            >
              <option value="">-- Chọn ban công --</option>
              <option value="Đông-Nam">Đông-Nam</option>
              <option value="Đông-Bắc">Đông-Bắc</option>
              <option value="Tây-Bắc">Tây-Bắc</option>
              <option value="Tây-Nam">Tây-Nam</option>
              <option value="Nam">Nam</option>
              <option value="Bắc">Bắc</option>
              <option value="Đông">Đông</option>
              <option value="Tây">Tây</option>
            </select>
          </div>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Đang xử lý..." : "Dự đoán giá"}
        </button>
      </form>

      {predictedPrice !== null && (
        <div className="result">
          <h3>Giá dự đoán: {predictedPrice}</h3>
        </div>
      )}
    </div>
  );
}

export default App;
