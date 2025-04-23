# Diccionarios de codificación creados a partir de los CSV _enumeration

# Marca del coche
brand_enum_dict = {
    'Acura': 0, 'Alfa': 1, 'Aston': 2, 'Audi': 3, 'BMW': 4, 'Bentley': 5, 'Bugatti': 6,
    'Buick': 7, 'Cadillac': 8, 'Chevrolet': 9, 'Chrysler': 10, 'Dodge': 11, 'FIAT': 12,
    'Ferrari': 13, 'Ford': 14, 'GMC': 15, 'Genesis': 16, 'Honda': 17, 'Hummer': 18,
    'Hyundai': 19, 'INFINITI': 20, 'Jaguar': 21, 'Jeep': 22, 'Karma': 23, 'Kia': 24,
    'Lamborghini': 25, 'Land': 26, 'Lexus': 27, 'Lincoln': 28, 'Lotus': 29, 'Lucid': 30,
    'MINI': 31, 'Maserati': 32, 'Maybach': 33, 'Mazda': 34, 'McLaren': 35,
    'Mercedes-Benz': 36, 'Mercury': 37, 'Mitsubishi': 38, 'Nissan': 39, 'Plymouth': 40,
    'Polestar': 41, 'Pontiac': 42, 'Porsche': 43, 'RAM': 44, 'Rivian': 45,
    'Rolls-Royce': 46, 'Saab': 47, 'Saturn': 48, 'Scion': 49, 'Subaru': 50, 'Suzuki': 51,
    'Tesla': 52, 'Toyota': 53, 'Volkswagen': 54, 'Volvo': 55, 'smart': 56
}

# Tipo de combustible
fuel_enum_dict = {
    'Diesel': 0, 'E85 Flex Fuel': 1, 'Electric': 2,
    'Gasoline': 3, 'Hybrid': 4, 'Plug-In Hybrid': 5
}

# Tipo de transmisión
transmission_enum_dict = {
    'Automatico': 0,
    'Manual': 1,
    'Otros': 2
}

def encode_brand(brand: str) -> int:
    """Convierte una marca en su código numérico según la enumeración."""
    return brand_enum_dict.get(brand, -1)

def encode_fuel_type(fuel: str) -> int:
    """Convierte un tipo de combustible en su código numérico según la enumeración."""
    return fuel_enum_dict.get(fuel, -1)

def encode_transmission(trans: str) -> int:
    """Convierte una transmisión en su código numérico según la enumeración."""
    return transmission_enum_dict.get(trans, 2)
