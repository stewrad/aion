# Optimization of Adaptive Modulation and Coding with Reinforcement Learning

## Satellite Communications

### IEEE Frequency Band Designation (EEE 521 Standard)
<div style="display: flex; gap: 40px;">
  <div style="flex: 2;">
    <h4>Band Name</h4>
    <ul>
      <li>HF</li>
      <li>VHF</li>
      <li>UHF</li>
      <li>L</li>
      <li>S</li>
      <li>C</li>
      <li>X</li>
      <li>Ku</li>
      <li>K</li>
      <li>Ka</li>
      <li>V</li>
      <li>W</li>
      <li>mm or G</li>
    </ul>
  </div>
  <div style="flex: 2;">
    <h4>Frequency Range</h4>
    <ul>
      <li>3 to 30 MHz</li>
      <li>30 to 300 MHz</li>
      <li>300 to 1000 MHz</li>
      <li>1 to 2 GHz</li>
      <li>2 to 4 Ghz</li>
      <li>4 to 8 GHz</li>
      <li>8 to 12 GHz</li>
      <li>12 to 18 GHz</li>
      <li>18 to 27 GHz</li>
      <li>27 to 40 GHz</li>
      <li>40 to 75 GHz</li>
      <li>75 to 110 GHz</li>
      <li>110 to 300 GHz</li>
    </ul>
  </div>
</div>

### Atmospheric Attenuation 

#### 1. Atmospheric Attenuation 
- **Rain fade** is the biggest seasonal channel effect, especially for frequencies **above 10 GHz** (Ku, Ka, and Q/V bands)
- During rainy seasons (Spring/Summer in many reigons), **signal power drops** due to **absoprtion** and **scattering** by the water droplets
- **Snow and ice crystals** also scatter energy, but generally less severely than rain 
- **Clouds** (especially thick cumulus and cumulonimbus) introduce additional attenuation in Ka-band links

**Example:**
A Ka-band downlink (20 GHz) in Florida experiences deep fades (up to 10–20 dB) in summer due to convective rain, while in winter it’s nearly negligible.

#### 2. Ionospheric Effects
The ionosphere varies with **solar activity, day/night cycles, and seasons**. This is most impactful for frequencies **below ~3GHz** (L and S bands)
- **Ionospheric delay** (group delay) and **scintillation** (rapid ampitude/phase fluctuations) vary with: 
     - **Season**: more intense in equinox periods (March and September) due to higher solar flux and geomagnetic activity 
     - **Latitude**: stronger near the equator and polar regions
     - **Solar cycle ($\approx$ 11 years)**: high solar activity increases as total electron content (TEC), worsening delay and fading 
    
**Example:**
GNSS (GPS) signals can have position errors up to 10–20 m during high TEC events in equinox seasons.

#### 3. Tropospheric Effects
The troposphere causes **refraction, scintillation, and path delay** that vary seasonally
- **Water vapor density**: higher in summer due to more refractive bending and variable path delay 
- **Temperature gradients** near the Earth's surface can cause **multipath or ducting** effects in low-elevation satellite links (e.g. LEO ground stations)

#### 4. Multipath & Ground Reflection Changes
- Seasonal vegetation changes (leaves, moisture content) alter **ground reflection characteristics**, slightly changing polarization and multipath interference patterns
- In maritime or coastal environments, **sea surface conditions** (calm vs rough) change reflection-induced fading throughout the year 

#### 5. Sun Outages (Solar Conjunctions)
- Occur **twice a year** (around the equinoxes) when the Sun passes directly behind the satellite relative to the Earth station 
- The Sun's radio noise temporarily overwhelms the satellite signal (C/N drops dramatically for a few minutes per day, for several days)

#### 6. Seasonal Pointing and Geometry Variations
 - Earth's **axial tilt** changes the apparent eleveation and azimuth and geostationary satellites slightly throughout the year 
 - This may affect **antenna gain** if not perfectly aligned, and it interacts with tropospheric and ionospheric path lengths 

### Satellite Communications at 6GHz+

#### 1. GEO Path Geometry
A GEO satellite link has a **very long slant path** (~35,786 km)
- Any atmospheric attenuation compounds over a long distance
- The **lowest few kilometers of atmosphere** (where weather happens) dominate the total path loss
- Because the geometry is fixed, **seasonal weather variations** directly translate to seasonal fade statistics

#### 2. Rain Attenuation - the Dominant Effect
Above **~10GHz**, the rain fade dominates all other impairments 

<div style="display: flex; gap: 40px;">
  <div style="flex: 4;">
    <h4>Band</h4>
    <ul>
      <li>C</li>
      <li>X</li>
      <li>Ku</li>
      <li>Ka</li>
      <li>Q/V</li>
    </ul>
  </div>
  <div style="flex: 4;">
    <h4>Frequency</h4>
    <ul>
      <li>4 to 8 GHz</li>
      <li>8 to 12 GHz</li>
      <li>12 to 18 GHz</li>
      <li>20 to 30 GHz</li>
      <li>40 to 75 GHz</li>
    </ul>
  </div>
  <div style="flex: 4;">
    <h4>Typical Use</h4>
    <ul>
      <li>Legacy TV, Military</li>
      <li>Military</li>
      <li>TV broadcast, VSAT</li>
      <li>High-throughput Satellite</li>
      <li>Experimental</li>
    </ul>
  </div>
  <div style="flex: 4;">
    <h4>Rain Attenuation Severity</h4>
    <ul>
      <li>Minor (0.1-1 dB)</li>
      <li>Manageable (1-3 dB)</li>
      <li>Moderate (3-10 dB)</li>
      <li>Severe (10-20 dB, >30 dB tropical rain)</li>
      <li>Extreme (40-60 dB fades possible)</li>
    </ul>
  </div>
</div>

### Engineering Countermeasures
To maintain **high link availability** (>99.9%), engineers use: 
- **Adaptive Coding and Modulation (ACM)**: Reduces data rate temporarily during fades
- **Uplink Power Control (UPC)**: Dynamically boosts transmit power
- **Diversity techniques**: Spatial (multiple gateways), frequency, or site diversity 
- **Fade margins**: Designing with an **extra 10-20 dB margin** at Ka-band


## Adapative Coding and Modulation (ACM)