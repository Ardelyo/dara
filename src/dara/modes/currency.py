"""
DARA Modes - Currency Detection Handler
Provides intelligent Indonesian Rupiah and multi-currency detection.
"""

import re
from .base import BaseMode, ModeResult
from ..utils.text import TextUtils


class CurrencyMode(BaseMode):
    """
    Currency detection mode for money identification.
    
    Specialized for Indonesian Rupiah with comprehensive
    denomination database including colors and features.
    """
    
    # Indonesian Rupiah denomination database
    IDR_DENOMINATIONS = {
        "100000": {
            "color_en": "red/pink",
            "color_id": "merah/pink",
            "figure": "Soekarno-Hatta",
            "value_text": "Rp 100.000",
            "keywords": ["100000", "100.000", "seratus ribu", "hundred thousand"]
        },
        "75000": {
            "color_en": "red-white",
            "color_id": "merah-putih",
            "figure": "Kemerdekaan",
            "value_text": "Rp 75.000",
            "keywords": ["75000", "75.000", "tujuh puluh lima ribu"]
        },
        "50000": {
            "color_en": "blue",
            "color_id": "biru",
            "figure": "I Gusti Ngurah Rai",
            "value_text": "Rp 50.000",
            "keywords": ["50000", "50.000", "lima puluh ribu", "fifty thousand"]
        },
        "20000": {
            "color_en": "green",
            "color_id": "hijau",
            "figure": "Otto Iskandar Dinata",
            "value_text": "Rp 20.000",
            "keywords": ["20000", "20.000", "dua puluh ribu", "twenty thousand"]
        },
        "10000": {
            "color_en": "purple",
            "color_id": "ungu",
            "figure": "Frans Kaisiepo",
            "value_text": "Rp 10.000",
            "keywords": ["10000", "10.000", "sepuluh ribu", "ten thousand"]
        },
        "5000": {
            "color_en": "brown",
            "color_id": "coklat",
            "figure": "Idham Chalid",
            "value_text": "Rp 5.000",
            "keywords": ["5000", "5.000", "lima ribu", "five thousand"]
        },
        "2000": {
            "color_en": "gray",
            "color_id": "abu-abu",
            "figure": "M. Hoesni Thamrin",
            "value_text": "Rp 2.000",
            "keywords": ["2000", "2.000", "dua ribu", "two thousand"]
        },
        "1000": {
            "color_en": "light green",
            "color_id": "hijau muda",
            "figure": "Tjut Meutia",
            "value_text": "Rp 1.000",
            "keywords": ["1000", "1.000", "seribu", "one thousand"]
        }
    }
    
    # Other currency patterns
    OTHER_CURRENCIES = {
        "USD": {"symbol": "$", "pattern": r'\$\s*[\d.,]+'},
        "EUR": {"symbol": "€", "pattern": r'€\s*[\d.,]+'},
        "GBP": {"symbol": "£", "pattern": r'£\s*[\d.,]+'},
    }
    
    @property
    def name(self) -> str:
        return self.MODE_CURRENCY
    
    @property
    def prompt(self) -> str:
        return "<OCR>"
    
    @property
    def description(self) -> str:
        return "Identifies Indonesian Rupiah and other currencies"
    
    def process(self, raw_output: str, language: str = "en") -> ModeResult:
        """
        Process currency detection output.
        
        Identifies:
        - Indonesian Rupiah denominations with colors
        - Other major currencies (USD, EUR, GBP)
        - Total value if multiple detected
        """
        text = self.preprocess(raw_output)
        normalized_text = TextUtils.normalize_currency(text)
        
        # Detect currencies
        idr_detected = self._detect_idr(normalized_text, text)
        other_detected = self._detect_other_currencies(text)
        
        patterns_matched = len(idr_detected) + len(other_detected)
        
        # Build output
        if idr_detected:
            output_text = self._format_idr_output(idr_detected, language)
        elif other_detected:
            output_text = self._format_other_output(other_detected, language)
        else:
            # Fallback: look for any numbers
            numbers = TextUtils.extract_numbers(text)
            if numbers:
                if language == "id":
                    output_text = f"Angka terdeteksi: {', '.join(numbers[:3])}"
                else:
                    output_text = f"Numbers detected: {', '.join(numbers[:3])}"
            else:
                output_text = "Mata uang tidak terdeteksi." if language == "id" else "Currency not detected."
        
        # Calculate confidence
        confidence = self.calculate_confidence(text, patterns_matched)
        if idr_detected:
            confidence = min(1.0, confidence + 0.2)  # Boost for IDR detection
        
        # Calculate total value
        total_idr = sum(int(d["denomination"]) for d in idr_detected)
        
        return ModeResult(
            text=output_text,
            confidence=confidence,
            raw_output=raw_output,
            metadata={
                "idr_detected": idr_detected,
                "other_detected": other_detected,
                "total_idr": total_idr,
                "count": len(idr_detected) + len(other_detected)
            },
            suggestions=self._get_suggestions(idr_detected, language)
        )
    
    def _detect_idr(self, normalized_text: str, original_text: str) -> list:
        """Detect Indonesian Rupiah denominations."""
        detected = []
        text_lower = normalized_text.lower()
        original_lower = original_text.lower()
        
        for denom, info in self.IDR_DENOMINATIONS.items():
            for keyword in info["keywords"]:
                if keyword in text_lower or keyword in original_lower:
                    detected.append({
                        "denomination": denom,
                        "value_text": info["value_text"],
                        "color_en": info["color_en"],
                        "color_id": info["color_id"],
                        "figure": info["figure"]
                    })
                    break
        
        # Also check for explicit Rp patterns
        rp_pattern = r'[Rr]p\.?\s*([\d.,]+)'
        for match in re.finditer(rp_pattern, normalized_text):
            value = match.group(1).replace('.', '').replace(',', '')
            if value in self.IDR_DENOMINATIONS:
                info = self.IDR_DENOMINATIONS[value]
                entry = {
                    "denomination": value,
                    "value_text": info["value_text"],
                    "color_en": info["color_en"],
                    "color_id": info["color_id"],
                    "figure": info["figure"]
                }
                if entry not in detected:
                    detected.append(entry)
        
        return detected
    
    def _detect_other_currencies(self, text: str) -> list:
        """Detect non-IDR currencies."""
        detected = []
        
        for currency, info in self.OTHER_CURRENCIES.items():
            matches = re.findall(info["pattern"], text)
            for match in matches:
                detected.append({
                    "currency": currency,
                    "value": match,
                    "symbol": info["symbol"]
                })
        
        return detected
    
    def _format_idr_output(self, detected: list, language: str) -> str:
        """Format output for IDR detection."""
        parts = []
        
        for item in detected:
            if language == "id":
                color = item["color_id"]
                parts.append(f"{item['value_text']} (warna {color})")
            else:
                color = item["color_en"]
                parts.append(f"{item['value_text']} ({color} color)")
        
        if len(detected) > 1:
            total = sum(int(d["denomination"]) for d in detected)
            if language == "id":
                return f"Terdeteksi: {', '.join(parts)}. Total: Rp {total:,}"
            else:
                return f"Detected: {', '.join(parts)}. Total: Rp {total:,}"
        else:
            if language == "id":
                return f"Terdeteksi: {parts[0]}"
            else:
                return f"Detected: {parts[0]}"
    
    def _format_other_output(self, detected: list, language: str) -> str:
        """Format output for non-IDR currencies."""
        parts = [f"{d['value']} ({d['currency']})" for d in detected]
        if language == "id":
            return f"Mata uang asing terdeteksi: {', '.join(parts)}"
        else:
            return f"Foreign currency detected: {', '.join(parts)}"
    
    def _get_suggestions(self, idr_detected: list, language: str) -> list:
        """Get contextual suggestions."""
        if language == "id":
            suggestions = ["Periksa ciri-ciri keamanan uang"]
            if idr_detected:
                suggestions.append(f"Warna utama: {idr_detected[0]['color_id']}")
        else:
            suggestions = ["Verify security features"]
            if idr_detected:
                suggestions.append(f"Primary color: {idr_detected[0]['color_en']}")
        
        return suggestions
