
SCENARIOS = {
    'scenario_1': {
        'id': 1,
        'name': 'ظهر تابستان - آفتابی و شلوغ',
        'z': {
            'T_out': 32,
            'H_out': 45,
            'Solar': 900,
            'Wind': 3,
            'N': 120,
            'W': 'sunny'
        },
        'category': 'hot'
    },

    'scenario_2': {
        'id': 2,
        'name': 'موج گرما - رطوبت کم',
        'z': {
            'T_out': 38,
            'H_out': 30,
            'Solar': 1000,
            'Wind': 2,
            'N': 80,
            'W': 'sunny'
        },
        'category': 'hot'
    },

    'scenario_3': {
        'id': 3,
        'name': 'هوای معتدل - ابری و پرتراکم',
        'z': {
            'T_out': 22,
            'H_out': 55,
            'Solar': 300,
            'Wind': 4,
            'N': 150,
            'W': 'cloudy'
        },
        'category': 'mild'
    },

    'scenario_4': {
        'id': 4,
        'name': 'بارانی - مرطوب',
        'z': {
            'T_out': 18,
            'H_out': 85,
            'Solar': 80,
            'Wind': 6,
            'N': 60,
            'W': 'rainy'
        },
        'category': 'wet'
    },

    'scenario_5': {
        'id': 5,
        'name': 'طوفانی',
        'z': {
            'T_out': 16,
            'H_out': 75,
            'Solar': 50,
            'Wind': 12,
            'N': 90,
            'W': 'stormy'
        },
        'category': 'storm'
    },

    'scenario_6': {
        'id': 6,
        'name': 'زمستان سرد',
        'z': {
            'T_out': 8,
            'H_out': 60,
            'Solar': 200,
            'Wind': 5,
            'N': 40,
            'W': 'cold'
        },
        'category': 'cold'
    },

    'scenario_7': {
        'id': 7,
        'name': 'سرد و بادی - شلوغ',
        'z': {
            'T_out': 3,
            'H_out': 70,
            'Solar': 100,
            'Wind': 8,
            'N': 110,
            'W': 'cold'
        },
        'category': 'cold'
    },

    'scenario_8': {
        'id': 8,
        'name': 'شرجی - مرطوب و گرم',
        'z': {
            'T_out': 28,
            'H_out': 90,
            'Solar': 600,
            'Wind': 2,
            'N': 130,
            'W': 'humid'
        },
        'category': 'humid'
    },

    'scenario_9': {
        'id': 9,
        'name': 'خشک - کم تراکم و خشک',
        'z': {
            'T_out': 26,
            'H_out': 20,
            'Solar': 700,
            'Wind': 5,
            'N': 30,
            'W': 'dry'
        },
        'category': 'dry'
    },

    'scenario_10': {
        'id': 10,
        'name': 'شب - بدون نور خورشید',
        'z': {
            'T_out': 20,
            'H_out': 65,
            'Solar': 0,
            'Wind': 4,
            'N': 100,
            'W': 'night'
        },
        'category': 'night'
    },
}

def get_all_scenarios():
    return SCENARIOS


def get_scenario(scenario_key: str):
    return SCENARIOS.get(scenario_key)


def get_scenario_by_id(scenario_id: int):
    for key, sc in SCENARIOS.items():
        if sc['id'] == scenario_id:
            return sc
    return None


def get_scenario_list():
    return list(SCENARIOS.keys())


def get_scenario_count():
    return len(SCENARIOS)


def get_scenarios_by_category(category: str):
    return {k: v for k, v in SCENARIOS.items() if v['category'] == category}


def print_scenarios_table():
    print("\n" + "=" * 90)
    print("Environmental Scenarios for Stage 3 (10 Scenarios)")
    print("=" * 90)

    print(f"\n{'ID':<4} {'Name':<30} {'T_out':<8} {'H_out':<8} {'Solar':<8} {'Wind':<8} {'N':<8} {'Weather':<10}")
    print("-" * 90)

    for key, sc in SCENARIOS.items():
        z = sc['z']
        print(f"{sc['id']:<4} "
              f"{sc['name'][:28]:<30} "
              f"{z['T_out']:<8} "
              f"{z['H_out']:<8} "
              f"{z['Solar']:<8} "
              f"{z['Wind']:<8} "
              f"{z['N']:<8} "
              f"{z['W']:<10}")

    print("\n" + "=" * 90)
    print(f"Total: {get_scenario_count()} scenarios")
    print("=" * 90)


def get_scenarios_for_team2():

    scenarios_list = []

    for key, sc in SCENARIOS.items():
        scenarios_list.append({
            'key': key,
            'id': sc['id'],
            'name': sc['name'],
            'z': sc['z'].copy(),
            'bounds': {
                'T_in': (10, 35),
                'H_in': (10, 90),
                'L': (0, 1000),
                'CO2': (300, 1700)
            }
        })

    return scenarios_list


def get_z_matrix():
    import numpy as np

    scenario_keys = get_scenario_list()
    z_matrix = []
    weather_list = []

    for key in scenario_keys:
        z = SCENARIOS[key]['z']
        z_matrix.append([
            z['T_out'],
            z['H_out'],
            z['Solar'],
            z['Wind'],
            z['N'],
        ])
        weather_list.append(z['W'])

    return np.array(z_matrix), weather_list


if __name__ == "__main__":
    print_scenarios_table()

