# Circle Analyzer

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/catalizcs/circle-analyzer/releases)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A sophisticated osu! replay analysis tool designed to detect gameplay patterns and validate authenticity through advanced statistical methods.

## Core Features

### Analysis Capabilities

- **Hold Pattern Analysis**: Advanced key press duration analytics
- **Frame Timing Analysis**: High-precision frame timing validation
- **Unstable Rate Analysis**: Statistical hit timing consistency evaluation
- **Pattern Recognition** (Beta): Algorithmic anomaly detection

### Technical Specifications

- Comprehensive multi-mod compatibility (DT/NC time scaling)
- Advanced statistical modeling and analysis
- Data visualization with matplotlib
- Efficient batch processing system
- Detailed technical reporting

## Installation

```bash
git clone https://github.com/catalizcs/circle-analyzer.git
cd circle-analyzer
pip install -r requirements.txt
```

## Usage Guide

### Basic Operation

```bash
python main.py
```

### Analysis Modes

1. **Comprehensive Analysis**: Full replay examination
2. **Hold Pattern Analysis**: Key press pattern evaluation
3. **Frame Timing Analysis**: Temporal distribution analysis
4. **UR Calculation**: Hit timing deviation metrics
5. **Pattern Recognition**: Algorithmic anomaly detection

### Batch Processing

Supports drag-and-drop `.osr` file processing:

```
h - Hold Pattern Analysis
f - Frame Timing Analysis
a - Comprehensive Analysis
c - Pattern Recognition
u - UR Calculation
```

## Configuration

System parameters can be configured via GUI or `settings.json`:

```json
{
  "output_directory": "output",
  "export_format": "png"
}
```

## Detection System (Beta)

The analyzer implements multiple detection algorithms:

- **Speed Validation**: Physical limitation verification
- **Frame Timing Analysis**: Statistical timing validation
- **Hold Pattern Analysis**: Key press consistency evaluation

⚠️ Detection systems are in beta phase and require human verification

## Output Specifications

### Generated Assets

- `*_holdtime.png`: Hold pattern visualization
- `*_frametime.png`: Timing distribution graph
- `*_ur.png`: Hit accuracy distribution
- `*_analysis.txt`: Technical analysis report

### Sample Report

```
Technical Analysis Report
=======================

[!] Pattern Analysis Results:
- SPEED_ANOMALY: Threshold exceeded [BETA]
    Confidence Rating: 0.95

Technical Metrics:
UR (Frame Timing): 75.647
Hold Pattern Distribution: K1=142.3ms, K2=194.0ms
```

## Technical Requirements

- Python 3.8+
- Dependencies:
  - matplotlib
  - numpy
  - osrparse
  - rich
  - inquirer

## Development

1. Fork repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Implement changes (`git commit -m 'Implement enhancement'`)
4. Push changes (`git push origin feature/enhancement`)
5. Submit Pull Request

## License

Licensed under MIT License - See [LICENSE](LICENSE)

## Credits

- osu! community and my friends for beta testing
- [osrparse](https://github.com/kszlim/osu-replay-parser) library

## Legal Notice

This software is provided for research and educational purposes. Analysis results should not be considered conclusive evidence.

## Developer

**catalizcs**

- [GitHub Profile](https://github.com/catalizcs)
