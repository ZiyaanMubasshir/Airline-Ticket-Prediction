from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# ── Load model and encoders ──
model                  = pickle.load(open("model.pkl", "rb"))
le_airline             = pickle.load(open("le_airline.pkl", "rb"))
le_source              = pickle.load(open("le_source.pkl", "rb"))
le_destination         = pickle.load(open("le_destination.pkl", "rb"))
le_additional          = pickle.load(open("le_additional.pkl", "rb"))
le_source_airport      = pickle.load(open("le_source_airport.pkl", "rb"))
le_destination_airport = pickle.load(open("le_destination_airport.pkl", "rb"))

# ── Build route lookup from dataset ──
def build_route_lookup():
    df = pd.read_csv("Data_Train.csv")
    df.dropna(inplace=True)

    def dur_to_min(d):
        h, m = 0, 0
        if 'h' in str(d): h = int(str(d).split('h')[0].strip())
        if 'm' in str(d): m = int(str(d).split('m')[0].split()[-1].strip())
        return h * 60 + m

    df['dur_mins'] = df['Duration'].apply(dur_to_min)
    stop_map = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    df['stops_num'] = df['Total_Stops'].map(stop_map).fillna(1)
    df['src_ap'] = df['Route'].apply(lambda x: x.split('?')[0].strip())
    df['dst_ap'] = df['Route'].apply(lambda x: x.split('?')[-1].strip())

    grp = df.groupby(['Airline', 'Source', 'Destination']).agg(
        avg_dur=('dur_mins', 'median'),
        avg_stops=('stops_num', 'median'),
        src_ap=('src_ap', lambda x: x.mode()[0]),
        dst_ap=('dst_ap', lambda x: x.mode()[0])
    ).reset_index()

    lookup = {}
    for _, row in grp.iterrows():
        key = (row['Airline'], row['Source'], row['Destination'])
        lookup[key] = {
            'dur_mins':  int(row['avg_dur']),
            'stops':     int(row['avg_stops']),
            'src_ap':    row['src_ap'],
            'dst_ap':    row['dst_ap'],
            'dur_hours': int(row['avg_dur']) // 60,
            'dur_min':   int(row['avg_dur']) % 60
        }
    return lookup

ROUTE_LOOKUP = build_route_lookup()

# ── Data for dropdowns ──
AIRLINES     = sorted(le_airline.classes_)
SOURCES      = sorted(le_source.classes_)
DESTINATIONS = sorted(le_destination.classes_)

# Valid routes set
VALID_ROUTES = {(k[0], k[1], k[2]) for k in ROUTE_LOOKUP.keys()}

def get_valid_destinations(airline, source):
    return sorted(set(d for (a, s, d) in VALID_ROUTES if a == airline and s == source))

def get_valid_sources(airline):
    return sorted(set(s for (a, s, d) in VALID_ROUTES if a == airline))


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict_page')
def predict_page():
    default_airline = AIRLINES[0]
    default_sources = get_valid_sources(default_airline)
    default_destinations = get_valid_destinations(default_airline, default_sources[0]) if default_sources else []
    return render_template("predict.html",
                           airlines=AIRLINES,
                           sources=default_sources,
                           destinations=default_destinations,
                           default_airline=default_airline)


@app.route('/get_sources')
def get_sources():
    airline = request.args.get('airline', '')
    sources = get_valid_sources(airline)
    return jsonify(sources)


@app.route('/get_destinations')
def get_destinations():
    airline = request.args.get('airline', '')
    source  = request.args.get('source', '')
    destinations = get_valid_destinations(airline, source)
    return jsonify(destinations)


@app.route('/get_route_info')
def get_route_info():
    airline = request.args.get('airline', '')
    source  = request.args.get('source', '')
    dest    = request.args.get('destination', '')
    key = (airline, source, dest)
    if key in ROUTE_LOOKUP:
        info = ROUTE_LOOKUP[key]
        return jsonify({
            'duration': f"{info['dur_hours']}h {info['dur_min']}m",
            'stops': info['stops'],
            'src_airport': info['src_ap'],
            'dst_airport': info['dst_ap']
        })
    return jsonify({'error': 'Route not found'})


def predict_price(airline, source, destination, date_str, dep_hour=12, dep_min=0):
    key = (airline, source, destination)
    if key not in ROUTE_LOOKUP:
        return None

    info = ROUTE_LOOKUP[key]
    dt   = datetime.strptime(date_str, "%Y-%m-%d")

    # Arrival time = departure + duration
    total_dep_mins = dep_hour * 60 + dep_min + info['dur_mins']
    arr_hour = (total_dep_mins // 60) % 24
    arr_min  = total_dep_mins % 60

    # Default additional info
    additional = 'No info'
    if additional not in le_additional.classes_:
        additional = le_additional.classes_[0]

    src_ap = info['src_ap']
    dst_ap = info['dst_ap']

    # Handle unknown airports
    if src_ap not in le_source_airport.classes_:
        src_ap = le_source_airport.classes_[0]
    if dst_ap not in le_destination_airport.classes_:
        dst_ap = le_destination_airport.classes_[0]

    features = np.array([[
        le_airline.transform([airline])[0],
        le_source.transform([source])[0],
        le_destination.transform([destination])[0],
        info['stops'],
        le_additional.transform([additional])[0],
        dt.day, dt.month, dt.year,
        dep_hour, dep_min,
        arr_hour, arr_min,
        info['dur_hours'], info['dur_min'], info['dur_mins'],
        le_source_airport.transform([src_ap])[0],
        le_destination_airport.transform([dst_ap])[0]
    ]])

    return round(float(model.predict(features)[0]), 2)


@app.route('/result', methods=['POST'])
def result():
    airline     = request.form.get('airline')
    source      = request.form.get('source')
    destination = request.form.get('destination')
    dep_date    = request.form.get('departure_date')
    dep_time    = request.form.get('dep_time', '12:00')
    trip_type   = request.form.get('trip_type', 'one')
    passengers  = int(request.form.get('passengers', 1))
    travel_class= request.form.get('travel_class', 'Economy')

    # Same city check
    if source.lower() == destination.lower():
        return render_template("predict.html",
                               error="Source and destination cannot be the same city!",
                               airlines=AIRLINES,
                               sources=get_valid_sources(airline),
                               destinations=get_valid_destinations(airline, source),
                               default_airline=airline)

    dep_h = int(dep_time.split(':')[0])
    dep_m = int(dep_time.split(':')[1])

    price = predict_price(airline, source, destination, dep_date, dep_h, dep_m)
    if price is None:
        return render_template("predict.html",
                               error=f"No route data found for {airline} from {source} to {destination}.",
                               airlines=AIRLINES,
                               sources=get_valid_sources(airline),
                               destinations=get_valid_destinations(airline, source),
                               default_airline=airline)

    # Class multiplier
    class_mult = {'Economy': 1.0, 'Business': 2.2, 'First Class': 3.5}
    mult = class_mult.get(travel_class, 1.0)
    price = round(price * mult, 2)

    total = round(price * passengers, 2)

    # Round trip
    return_total = None
    if trip_type == 'round':
        ret_date = request.form.get('return_date', dep_date)
        ret_price = predict_price(airline, destination, source, ret_date, dep_h, dep_m)
        if ret_price:
            ret_price = round(ret_price * mult, 2)
            return_total = round(ret_price * passengers, 2)
            total = round(total + return_total, 2)

    # Route info for display
    key = (airline, source, destination)
    info = ROUTE_LOOKUP.get(key, {})
    duration_str = f"{info.get('dur_hours', 0)}h {info.get('dur_min', 0)}m"
    stops_str    = f"{info.get('stops', 1)} stop(s)" if info.get('stops', 1) > 0 else "Non-stop"

    return render_template("result.html",
                           price=f"{price:,.2f}",
                           total=f"{total:,.2f}",
                           return_price=f"{return_total:,.2f}" if return_total else None,
                           passengers=passengers,
                           airline=airline,
                           source=source,
                           destination=destination,
                           dep_date=dep_date,
                           dep_time=dep_time,
                           travel_class=travel_class,
                           duration=duration_str,
                           stops=stops_str,
                           trip_type=trip_type)


if __name__ == "__main__":
    app.run(debug=True)
