"""
AutoGen Vehicle Knowledge Base Builder
Compiles a comprehensive static vehicle database from multiple free sources.
Output: vehicleKnowledgeBase.json — make, model, variant, category per entry.
No years — AI handles year context in knowledge cards.

Run: python3 buildVehicleKB.py
Output written to: autogen-backend/data/vehicleKnowledgeBase.json

Sources used:
1. NHTSA vPIC API — US market vehicles, very comprehensive
2. Wikipedia category pages — aircraft, ships, spacecraft, military
3. Hardcoded curated lists — specialty vehicles, motorcycles, heavy machinery
"""

import requests
import json
import time
import os
from collections import defaultdict

OUTPUT_PATH = "autogen-backend/data/vehicleKnowledgeBase.json"
HEADERS = {"User-Agent": "AutoGen FYP Vehicle KB Builder/1.0 (rafay@rafay.pro)"}

entries = []
seen = set()

def add(make, model, variant="", category="Land", vehicle_type=""):
    """Add an entry if not already seen."""
    key = f"{make.lower()}|{model.lower()}|{variant.lower()}"
    if key in seen:
        return
    seen.add(key)
    entries.append({
        "make": make,
        "model": model,
        "variant": variant,
        "category": category,
        "type": vehicle_type,
    })

def log(msg):
    print(msg, flush=True)

# ─────────────────────────────────────────────────────────────────
# SOURCE 1: NHTSA vPIC — all US market makes and models
# Free, no key, extremely comprehensive for cars and trucks
# ─────────────────────────────────────────────────────────────────

def fetch_nhtsa():
    log("\n[1/4] Fetching NHTSA vPIC vehicle data...")
    
    # Get all makes
    try:
        r = requests.get(
            "https://vpic.nhtsa.dot.gov/api/vehicles/getallmakes?format=json",
            headers=HEADERS, timeout=30
        )
        makes_data = r.json().get("Results", [])
        log(f"    Found {len(makes_data)} makes from NHTSA")
    except Exception as e:
        log(f"    NHTSA makes fetch failed: {e}")
        return

    # For each make get models — process top makes to avoid rate limits
    # NHTSA has ~11,000 makes including obscure ones
    # We want vehicle types: Passenger Car, Multipurpose Passenger Vehicle (MPV),
    # Truck, Motorcycle, Bus, Low Speed Vehicle
    
    vehicle_types = ["Passenger Car", "Truck", "Multipurpose Passenger Vehicle (MPV)", 
                     "Motorcycle", "Bus", "Incomplete Vehicle"]
    
    processed = 0
    for make_entry in makes_data:
        make_id = make_entry.get("Make_ID")
        make_name = make_entry.get("Make_Name", "").strip()
        
        if not make_name or len(make_name) < 2:
            continue
            
        try:
            r = requests.get(
                f"https://vpic.nhtsa.dot.gov/api/vehicles/getmodelsformakeid/{make_id}?format=json",
                headers=HEADERS, timeout=15
            )
            models = r.json().get("Results", [])
            
            for m in models:
                model_name = m.get("Model_Name", "").strip()
                if not model_name:
                    continue
                
                # Determine category from make name hints
                make_lower = make_name.lower()
                if any(x in make_lower for x in ["aircraft", "aviation", "aero", "boeing", "airbus", "cessna"]):
                    cat = "Air"
                elif any(x in make_lower for x in ["marine", "yacht", "boat", "sea"]):
                    cat = "Sea"
                else:
                    cat = "Land"
                
                add(make_name, model_name, "", cat)
            
            processed += 1
            if processed % 100 == 0:
                log(f"    Processed {processed}/{len(makes_data)} makes, {len(entries)} entries so far")
            
            time.sleep(0.05)  # gentle rate limiting
            
        except Exception:
            continue
    
    log(f"    NHTSA complete: {len(entries)} entries")


# ─────────────────────────────────────────────────────────────────
# SOURCE 2: CarQuery API — global vehicles 1941-present
# Free, no key, good for international makes
# Note: must be called server-side (no CORS)
# ─────────────────────────────────────────────────────────────────

def fetch_carquery():
    log("\n[2/4] Fetching CarQuery global vehicle data...")
    
    try:
        # Get all years available
        r = requests.get(
            "http://www.carqueryapi.com/api/0.3/?cmd=getYears",
            headers=HEADERS, timeout=15
        )
        # CarQuery returns JSONP, strip the callback wrapper
        text = r.text.strip()
        if text.startswith("?({"):
            text = text[2:-1]
        elif "({" in text:
            text = text[text.index("({")+1:text.rindex(")")]
        
        years_data = json.loads(text)
        min_year = years_data.get("Years", {}).get("min_year", 1990)
        max_year = years_data.get("Years", {}).get("max_year", 2024)
        log(f"    CarQuery years: {min_year} to {max_year}")
    except Exception as e:
        log(f"    CarQuery years fetch failed: {e}")
        return

    # Get makes
    try:
        r = requests.get(
            "http://www.carqueryapi.com/api/0.3/?cmd=getMakes",
            headers=HEADERS, timeout=15
        )
        text = r.text.strip()
        if "({" in text:
            text = text[text.index("({")+1:text.rindex(")")]
        makes_data = json.loads(text).get("Makes", [])
        log(f"    Found {len(makes_data)} makes from CarQuery")
    except Exception as e:
        log(f"    CarQuery makes fetch failed: {e}")
        return

    # Get models for each make
    processed = 0
    for make_entry in makes_data:
        make_id = make_entry.get("make_id", "")
        make_display = make_entry.get("make_display", "").strip()
        
        if not make_display:
            continue
        
        try:
            r = requests.get(
                f"http://www.carqueryapi.com/api/0.3/?cmd=getModels&make={make_id}",
                headers=HEADERS, timeout=15
            )
            text = r.text.strip()
            if "({" in text:
                text = text[text.index("({")+1:text.rindex(")")]
            models = json.loads(text).get("Models", [])
            
            for m in models:
                model_name = m.get("model_name", "").strip()
                if model_name:
                    add(make_display, model_name, "", "Land")
            
            processed += 1
            if processed % 50 == 0:
                log(f"    Processed {processed}/{len(makes_data)} makes")
            
            time.sleep(0.1)
            
        except Exception:
            continue
    
    log(f"    CarQuery complete: {len(entries)} total entries")


# ─────────────────────────────────────────────────────────────────
# SOURCE 3: Wikipedia category scraping for non-car vehicles
# Aircraft, ships, spacecraft, military vehicles
# ─────────────────────────────────────────────────────────────────

def fetch_wikipedia_category(category_title, make_hint, model_hint, cat, vtype):
    """Fetch members of a Wikipedia category."""
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category_title}",
                "cmlimit": 500,
                "format": "json"
            },
            headers=HEADERS,
            timeout=15
        )
        members = r.json().get("query", {}).get("categorymembers", [])
        for m in members:
            title = m.get("title", "").strip()
            if not title or ":" in title:
                continue
            # Try to parse make and model from title
            parts = title.split(" ", 1)
            if len(parts) == 2:
                add(parts[0], parts[1], "", cat, vtype)
            else:
                add(make_hint, title, "", cat, vtype)
        return len(members)
    except Exception as e:
        log(f"    Wikipedia category {category_title} failed: {e}")
        return 0


def fetch_wikipedia():
    log("\n[3/4] Fetching Wikipedia vehicle categories...")
    
    categories = [
        # Aircraft
        ("Military aircraft of the United States", "USAF", "", "Air", "Military Aircraft"),
        ("Boeing aircraft", "Boeing", "", "Air", "Commercial Aircraft"),
        ("Airbus aircraft", "Airbus", "", "Air", "Commercial Aircraft"),
        ("Cessna aircraft", "Cessna", "", "Air", "General Aviation"),
        ("Lockheed aircraft", "Lockheed", "", "Air", "Military Aircraft"),
        ("McDonnell Douglas aircraft", "McDonnell Douglas", "", "Air", "Commercial Aircraft"),
        ("Helicopter models", "", "", "Air", "Helicopter"),
        ("Unmanned aerial vehicles", "", "", "Air", "UAV"),
        
        # Ships and maritime
        ("Aircraft carriers", "", "", "Sea", "Military"),
        ("Submarines", "", "", "Sea", "Military"),
        ("Ocean liners", "", "", "Sea", "Commercial"),
        ("Cruise ships", "", "", "Sea", "Commercial"),
        ("Superyachts", "", "", "Sea", "Recreational"),
        
        # Spacecraft
        ("Rockets", "", "", "Space", "Rocket"),
        ("Space probes", "NASA", "", "Space", "Probe"),
        ("Crewed spacecraft", "", "", "Space", "Crewed"),
        ("Lunar rovers", "NASA", "", "Space", "Rover"),
        
        # Military land vehicles
        ("Main battle tanks", "", "", "Land", "Military"),
        ("Armoured personnel carriers", "", "", "Land", "Military"),
        ("Military trucks", "", "", "Land", "Military"),
        
        # Motorcycles
        ("Harley-Davidson motorcycles", "Harley-Davidson", "", "Land", "Motorcycle"),
        ("Ducati motorcycles", "Ducati", "", "Land", "Motorcycle"),
        ("BMW motorcycles", "BMW", "", "Land", "Motorcycle"),
        ("Honda motorcycles", "Honda", "", "Land", "Motorcycle"),
        ("Yamaha motorcycles", "Yamaha", "", "Land", "Motorcycle"),
        ("Kawasaki motorcycles", "Kawasaki", "", "Land", "Motorcycle"),
        ("Suzuki motorcycles", "Suzuki", "", "Land", "Motorcycle"),
        ("Royal Enfield motorcycles", "Royal Enfield", "", "Land", "Motorcycle"),
        ("Triumph motorcycles", "Triumph", "", "Land", "Motorcycle"),
        ("KTM motorcycles", "KTM", "", "Land", "Motorcycle"),
    ]
    
    for cat_title, make, model, category, vtype in categories:
        count = fetch_wikipedia_category(cat_title, make, model, category, vtype)
        log(f"    {cat_title}: {count} entries")
        time.sleep(0.3)
    
    log(f"    Wikipedia complete: {len(entries)} total entries")


# ─────────────────────────────────────────────────────────────────
# SOURCE 4: Hardcoded curated lists
# High-value vehicles that might not appear in automated sources
# Exotics, classics, specialty, aircraft variants, famous ships
# ─────────────────────────────────────────────────────────────────

CURATED = [
    # ── EXOTIC CARS ──
    ("Lamborghini", "Aventador", "LP700-4", "Land", "Supercar"),
    ("Lamborghini", "Aventador", "LP720-4", "Land", "Supercar"),
    ("Lamborghini", "Aventador", "S", "Land", "Supercar"),
    ("Lamborghini", "Aventador", "SVJ", "Land", "Supercar"),
    ("Lamborghini", "Aventador", "Ultimae", "Land", "Supercar"),
    ("Lamborghini", "Huracán", "LP610-4", "Land", "Supercar"),
    ("Lamborghini", "Huracán", "EVO", "Land", "Supercar"),
    ("Lamborghini", "Huracán", "STO", "Land", "Supercar"),
    ("Lamborghini", "Huracán", "Tecnica", "Land", "Supercar"),
    ("Lamborghini", "Urus", "S", "Land", "SUV"),
    ("Lamborghini", "Urus", "Performante", "Land", "SUV"),
    ("Lamborghini", "Revuelto", "", "Land", "Supercar"),
    ("Lamborghini", "Countach", "LPI 800-4", "Land", "Supercar"),
    ("Lamborghini", "Miura", "P400", "Land", "Supercar"),
    ("Lamborghini", "Gallardo", "LP560-4", "Land", "Supercar"),
    ("Ferrari", "F40", "", "Land", "Supercar"),
    ("Ferrari", "F50", "", "Land", "Supercar"),
    ("Ferrari", "Enzo", "", "Land", "Hypercar"),
    ("Ferrari", "LaFerrari", "", "Land", "Hypercar"),
    ("Ferrari", "SF90 Stradale", "", "Land", "Hypercar"),
    ("Ferrari", "488 GTB", "", "Land", "Supercar"),
    ("Ferrari", "488 Pista", "", "Land", "Supercar"),
    ("Ferrari", "F8 Tributo", "", "Land", "Supercar"),
    ("Ferrari", "Roma", "", "Land", "GT"),
    ("Ferrari", "Portofino", "M", "Land", "GT"),
    ("Ferrari", "812 Superfast", "", "Land", "GT"),
    ("Ferrari", "812 GTS", "", "Land", "GT"),
    ("Ferrari", "296 GTB", "", "Land", "Supercar"),
    ("Ferrari", "Purosangue", "", "Land", "SUV"),
    ("Ferrari", "Daytona SP3", "", "Land", "Hypercar"),
    ("Pagani", "Zonda", "F", "Land", "Hypercar"),
    ("Pagani", "Zonda", "R", "Land", "Hypercar"),
    ("Pagani", "Zonda", "Cinque", "Land", "Hypercar"),
    ("Pagani", "Huayra", "", "Land", "Hypercar"),
    ("Pagani", "Huayra", "BC", "Land", "Hypercar"),
    ("Pagani", "Huayra", "Roadster", "Land", "Hypercar"),
    ("Pagani", "Utopia", "", "Land", "Hypercar"),
    ("Koenigsegg", "Agera", "RS", "Land", "Hypercar"),
    ("Koenigsegg", "Jesko", "", "Land", "Hypercar"),
    ("Koenigsegg", "Jesko", "Absolut", "Land", "Hypercar"),
    ("Koenigsegg", "Gemera", "", "Land", "Hypercar"),
    ("Koenigsegg", "CC850", "", "Land", "Hypercar"),
    ("Bugatti", "Veyron", "16.4", "Land", "Hypercar"),
    ("Bugatti", "Veyron", "Super Sport", "Land", "Hypercar"),
    ("Bugatti", "Chiron", "", "Land", "Hypercar"),
    ("Bugatti", "Chiron", "Super Sport 300+", "Land", "Hypercar"),
    ("Bugatti", "Chiron", "Pur Sport", "Land", "Hypercar"),
    ("Bugatti", "Bolide", "", "Land", "Hypercar"),
    ("Bugatti", "Mistral", "", "Land", "Hypercar"),
    ("McLaren", "F1", "", "Land", "Hypercar"),
    ("McLaren", "P1", "", "Land", "Hypercar"),
    ("McLaren", "Senna", "", "Land", "Hypercar"),
    ("McLaren", "720S", "", "Land", "Supercar"),
    ("McLaren", "765LT", "", "Land", "Supercar"),
    ("McLaren", "750S", "", "Land", "Supercar"),
    ("McLaren", "Artura", "", "Land", "Supercar"),
    ("McLaren", "Elva", "", "Land", "Hypercar"),
    ("McLaren", "Solus GT", "", "Land", "Hypercar"),
    ("Porsche", "918 Spyder", "", "Land", "Hypercar"),
    ("Porsche", "911 GT3", "RS", "Land", "Supercar"),
    ("Porsche", "911 GT2", "RS", "Land", "Supercar"),
    ("Porsche", "911 Turbo", "S", "Land", "Supercar"),
    ("Porsche", "Taycan", "Turbo S", "Land", "Electric"),
    ("Porsche", "Cayenne", "Turbo GT", "Land", "SUV"),
    ("Aston Martin", "Valkyrie", "", "Land", "Hypercar"),
    ("Aston Martin", "Vantage", "AMR", "Land", "GT"),
    ("Aston Martin", "DBS", "Superleggera", "Land", "GT"),
    ("Aston Martin", "DB11", "AMR", "Land", "GT"),
    ("Aston Martin", "DBX", "707", "Land", "SUV"),
    ("Aston Martin", "One-77", "", "Land", "Hypercar"),
    ("Rolls-Royce", "Phantom", "VIII", "Land", "Luxury"),
    ("Rolls-Royce", "Ghost", "Extended", "Land", "Luxury"),
    ("Rolls-Royce", "Cullinan", "Black Badge", "Land", "SUV"),
    ("Rolls-Royce", "Spectre", "", "Land", "Electric"),
    ("Rolls-Royce", "Wraith", "Black Badge", "Land", "Luxury"),
    ("Bentley", "Continental GT", "Speed", "Land", "GT"),
    ("Bentley", "Flying Spur", "W12", "Land", "Luxury"),
    ("Bentley", "Bentayga", "EWB", "Land", "SUV"),
    ("Bentley", "Mulsanne", "Speed", "Land", "Luxury"),
    
    # ── CLASSIC CARS ──
    ("Ford", "Mustang", "Shelby GT500", "Land", "Muscle Car"),
    ("Ford", "Mustang", "Boss 429", "Land", "Muscle Car"),
    ("Ford", "GT40", "Mk IV", "Land", "Racing"),
    ("Ford", "Model T", "", "Land", "Classic"),
    ("Chevrolet", "Camaro", "ZL1 1LE", "Land", "Muscle Car"),
    ("Chevrolet", "Corvette", "C8 Z06", "Land", "Supercar"),
    ("Dodge", "Challenger", "SRT Demon", "Land", "Muscle Car"),
    ("Dodge", "Viper", "ACR", "Land", "Supercar"),
    ("Volkswagen", "Beetle", "1200", "Land", "Classic"),
    ("Mercedes-Benz", "300 SL", "Gullwing", "Land", "Classic"),
    ("Jaguar", "E-Type", "Series 1", "Land", "Classic"),
    ("Alfa Romeo", "Giulia", "GTA", "Land", "Sports"),
    
    # ── ELECTRIC ──
    ("Tesla", "Model S", "Plaid", "Land", "Electric"),
    ("Tesla", "Model 3", "Performance", "Land", "Electric"),
    ("Tesla", "Model X", "Plaid", "Land", "Electric"),
    ("Tesla", "Model Y", "Performance", "Land", "Electric"),
    ("Tesla", "Cybertruck", "Cyberbeast", "Land", "Electric"),
    ("Tesla", "Roadster", "", "Land", "Electric"),
    ("Rivian", "R1T", "", "Land", "Electric"),
    ("Rivian", "R1S", "", "Land", "Electric"),
    ("Lucid", "Air", "Grand Touring", "Land", "Electric"),
    ("NIO", "ET9", "", "Land", "Electric"),
    ("BYD", "Han", "EV", "Land", "Electric"),
    
    # ── MOTORCYCLES ──
    ("Ducati", "Panigale V4", "S", "Land", "Motorcycle"),
    ("Ducati", "Panigale V4", "R", "Land", "Motorcycle"),
    ("Ducati", "Panigale V4", "SP2", "Land", "Motorcycle"),
    ("Ducati", "Streetfighter V4", "S", "Land", "Motorcycle"),
    ("Ducati", "Multistrada V4", "S", "Land", "Motorcycle"),
    ("Ducati", "Diavel V4", "", "Land", "Motorcycle"),
    ("Ducati", "Monster", "SP", "Land", "Motorcycle"),
    ("Yamaha", "YZF-R1", "M", "Land", "Motorcycle"),
    ("Yamaha", "YZF-R6", "", "Land", "Motorcycle"),
    ("Yamaha", "MT-10", "SP", "Land", "Motorcycle"),
    ("Yamaha", "VMAX", "", "Land", "Motorcycle"),
    ("Honda", "CBR1000RR-R", "Fireblade SP", "Land", "Motorcycle"),
    ("Honda", "CB1000R", "Black Edition", "Land", "Motorcycle"),
    ("Honda", "Gold Wing", "Tour", "Land", "Motorcycle"),
    ("Honda", "Africa Twin", "Adventure Sports", "Land", "Motorcycle"),
    ("Kawasaki", "Ninja ZX-10R", "SE", "Land", "Motorcycle"),
    ("Kawasaki", "Ninja H2", "R", "Land", "Motorcycle"),
    ("Kawasaki", "Z900", "RS", "Land", "Motorcycle"),
    ("Kawasaki", "W800", "", "Land", "Motorcycle"),
    ("BMW", "S1000RR", "M", "Land", "Motorcycle"),
    ("BMW", "R 1250 GS", "Adventure", "Land", "Motorcycle"),
    ("BMW", "M1000RR", "", "Land", "Motorcycle"),
    ("Harley-Davidson", "Fat Boy", "114", "Land", "Motorcycle"),
    ("Harley-Davidson", "Street Glide", "Special", "Land", "Motorcycle"),
    ("Harley-Davidson", "Sportster", "S", "Land", "Motorcycle"),
    ("Harley-Davidson", "Pan America", "1250", "Land", "Motorcycle"),
    ("Royal Enfield", "Bullet", "350", "Land", "Motorcycle"),
    ("Royal Enfield", "Continental GT", "650", "Land", "Motorcycle"),
    ("Royal Enfield", "Himalayan", "452", "Land", "Motorcycle"),
    ("Triumph", "Bonneville", "T120", "Land", "Motorcycle"),
    ("Triumph", "Street Triple", "RS", "Land", "Motorcycle"),
    ("Triumph", "Tiger", "900 Rally Pro", "Land", "Motorcycle"),
    ("KTM", "1290 Super Duke", "R EVO", "Land", "Motorcycle"),
    ("KTM", "890 Adventure", "R", "Land", "Motorcycle"),
    ("MV Agusta", "F4", "RC", "Land", "Motorcycle"),
    ("MV Agusta", "Brutale", "1000 RR", "Land", "Motorcycle"),
    ("Indian", "Scout", "Bobber", "Land", "Motorcycle"),
    ("Indian", "Challenger", "Dark Horse", "Land", "Motorcycle"),
    
    # ── AIRCRAFT ──
    ("Boeing", "747", "400", "Air", "Commercial"),
    ("Boeing", "747", "8", "Air", "Commercial"),
    ("Boeing", "737", "MAX 10", "Air", "Commercial"),
    ("Boeing", "777", "X", "Air", "Commercial"),
    ("Boeing", "787", "Dreamliner", "Air", "Commercial"),
    ("Boeing", "B-2", "Spirit", "Air", "Military"),
    ("Boeing", "B-52", "Stratofortress", "Air", "Military"),
    ("Boeing", "F/A-18", "Super Hornet", "Air", "Military"),
    ("Airbus", "A380", "800", "Air", "Commercial"),
    ("Airbus", "A350", "XWB", "Air", "Commercial"),
    ("Airbus", "A220", "300", "Air", "Commercial"),
    ("Airbus", "A320neo", "", "Air", "Commercial"),
    ("Concorde", "Type 1", "", "Air", "Commercial"),
    ("Lockheed Martin", "F-22 Raptor", "", "Air", "Military"),
    ("Lockheed Martin", "F-35", "Lightning II", "Air", "Military"),
    ("Lockheed Martin", "SR-71", "Blackbird", "Air", "Military"),
    ("Lockheed Martin", "C-130", "Hercules", "Air", "Military"),
    ("Lockheed Martin", "U-2", "", "Air", "Military"),
    ("Northrop Grumman", "B-21 Raider", "", "Air", "Military"),
    ("Northrop Grumman", "B-2 Spirit", "", "Air", "Military"),
    ("Sukhoi", "Su-57", "", "Air", "Military"),
    ("Sukhoi", "Su-27", "", "Air", "Military"),
    ("Mikoyan", "MiG-29", "", "Air", "Military"),
    ("Mikoyan", "MiG-21", "", "Air", "Military"),
    ("Eurofighter", "Typhoon", "", "Air", "Military"),
    ("Dassault", "Rafale", "C", "Air", "Military"),
    ("Dassault", "Mirage 2000", "", "Air", "Military"),
    ("Cessna", "172 Skyhawk", "", "Air", "General Aviation"),
    ("Cessna", "Citation", "Latitude", "Air", "Business Jet"),
    ("Piper", "PA-28 Cherokee", "", "Air", "General Aviation"),
    ("Piper", "M600", "", "Air", "General Aviation"),
    ("Beechcraft", "King Air", "350", "Air", "Turboprop"),
    ("Gulfstream", "G700", "", "Air", "Business Jet"),
    ("Gulfstream", "G650ER", "", "Air", "Business Jet"),
    ("Bombardier", "Global 7500", "", "Air", "Business Jet"),
    ("Embraer", "Phenom 300E", "", "Air", "Business Jet"),
    ("Bell", "407", "", "Air", "Helicopter"),
    ("Sikorsky", "UH-60 Black Hawk", "", "Air", "Helicopter"),
    ("Boeing", "CH-47 Chinook", "", "Air", "Helicopter"),
    ("DJI", "Matrice 300 RTK", "", "Air", "UAV"),
    ("DJI", "Agras T40", "", "Air", "UAV"),
    ("General Atomics", "MQ-9 Reaper", "", "Air", "Military UAV"),
    ("General Atomics", "MQ-1 Predator", "", "Air", "Military UAV"),
    ("Northrop Grumman", "RQ-4 Global Hawk", "", "Air", "Military UAV"),
    ("Supermarine", "Spitfire", "Mk IX", "Air", "Military"),
    ("North American", "P-51 Mustang", "", "Air", "Military"),
    ("de Havilland", "Mosquito", "", "Air", "Military"),
    
    # ── SEA ──
    ("Sunseeker", "Predator", "57", "Sea", "Motor Yacht"),
    ("Sunseeker", "Manhattan", "68", "Sea", "Motor Yacht"),
    ("Azimut", "Grande", "35 Metri", "Sea", "Superyacht"),
    ("Ferretti", "920", "", "Sea", "Motor Yacht"),
    ("Riva", "Aquarama", "", "Sea", "Classic"),
    ("Beneteau", "Oceanis", "46.1", "Sea", "Sailing Yacht"),
    ("Jeanneau", "Sun Odyssey", "440", "Sea", "Sailing Yacht"),
    ("Hobie Cat", "16", "", "Sea", "Catamaran"),
    ("Laser", "ILCA 7", "", "Sea", "Dinghy"),
    ("Yamaha", "WaveRunner FX SVHO", "", "Sea", "Jet Ski"),
    ("Sea-Doo", "RXP-X 325", "", "Sea", "Jet Ski"),
    ("Boston Whaler", "370 Outrage", "", "Sea", "Fishing"),
    ("Maersk", "Triple-E", "", "Sea", "Container Ship"),
    ("Queen Mary", "2", "", "Sea", "Ocean Liner"),
    ("USS Gerald R. Ford", "CVN-78", "", "Sea", "Aircraft Carrier"),
    ("USS Zumwalt", "DDG-1000", "", "Sea", "Destroyer"),
    ("Virginia", "class submarine", "", "Sea", "Submarine"),
    ("Triton", "36000/2", "", "Sea", "Submersible"),
    
    # ── SPACE ──
    ("SpaceX", "Falcon 9", "Block 5", "Space", "Rocket"),
    ("SpaceX", "Falcon Heavy", "", "Space", "Rocket"),
    ("SpaceX", "Starship", "", "Space", "Rocket"),
    ("SpaceX", "Dragon", "Crew", "Space", "Spacecraft"),
    ("NASA", "Space Shuttle", "Orbiter", "Space", "Spacecraft"),
    ("NASA", "Lunar Roving Vehicle", "", "Space", "Rover"),
    ("NASA", "Perseverance", "", "Space", "Rover"),
    ("NASA", "Curiosity", "", "Space", "Rover"),
    ("NASA", "Artemis", "SLS", "Space", "Rocket"),
    ("Blue Origin", "New Shepard", "", "Space", "Rocket"),
    ("Blue Origin", "New Glenn", "", "Space", "Rocket"),
    ("Roscosmos", "Soyuz", "MS", "Space", "Spacecraft"),
    ("ESA", "Ariane 5", "", "Space", "Rocket"),
    ("ESA", "Ariane 6", "", "Space", "Rocket"),
    
    # ── HEAVY MACHINERY ──
    ("Caterpillar", "797F", "", "Land", "Mining Truck"),
    ("Caterpillar", "D11", "", "Land", "Bulldozer"),
    ("Caterpillar", "390F", "", "Land", "Excavator"),
    ("John Deere", "8R 410", "", "Land", "Tractor"),
    ("John Deere", "9RX 640", "", "Land", "Tractor"),
    ("Liebherr", "LTM 11200-9.1", "", "Land", "Crane"),
    ("Kenworth", "W900", "", "Land", "Semi Truck"),
    ("Peterbilt", "389", "", "Land", "Semi Truck"),
    ("Freightliner", "Cascadia", "", "Land", "Semi Truck"),
    ("Zamboni", "552", "", "Land", "Ice Resurfacer"),
    ("Mercedes-Benz", "Unimog", "U4023", "Land", "Utility"),
    ("Land Rover", "Defender", "Works V8", "Land", "Off-Road"),
    
    # ── MILITARY LAND ──
    ("Leopard", "2A7+", "", "Land", "Main Battle Tank"),
    ("M1 Abrams", "SEPv3", "", "Land", "Main Battle Tank"),
    ("T-14 Armata", "", "", "Land", "Main Battle Tank"),
    ("Challenger", "2 LEP", "", "Land", "Main Battle Tank"),
    ("Humvee", "M1151A1", "", "Land", "Military Vehicle"),
    ("Land Rover", "Defender Wolf", "", "Land", "Military Vehicle"),
]


def add_curated():
    log("\n[4/4] Adding curated high-value vehicles...")
    before = len(entries)
    for item in CURATED:
        add(*item)
    log(f"    Added {len(entries) - before} curated entries")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("=" * 60)
    log("AutoGen Vehicle Knowledge Base Builder")
    log("=" * 60)

    # Run all sources
    fetch_nhtsa()
    fetch_carquery()
    fetch_wikipedia()
    add_curated()

    # Deduplicate and sort
    log(f"\nTotal entries before dedup: {len(entries)}")
    
    # Final sort: category, make, model, variant
    entries.sort(key=lambda x: (
        x["category"], 
        x["make"].lower(), 
        x["model"].lower(), 
        x["variant"].lower()
    ))

    log(f"Final entries: {len(entries)}")

    # Category breakdown
    from collections import Counter
    cats = Counter(e["category"] for e in entries)
    log("\nBy category:")
    for cat, count in sorted(cats.items()):
        log(f"  {cat}: {count:,}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    log(f"\nSaved to {OUTPUT_PATH}")
    log(f"File size: {size_kb:.1f} KB")
    log("\nDone.")
