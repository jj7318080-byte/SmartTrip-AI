
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash, session
)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
from pathlib import Path
from sqlalchemy import text
import joblib
import pandas as pd
import json
import math

# ==========================================
# 1. APPLICATION SETUP
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

app.config["SECRET_KEY"] = "smarttrip-local-demo-secret"
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + str(INSTANCE_DIR / "smarttrip.db")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
# ==========================================
# LOGIN PROTECTION
# ==========================================

def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to continue.", "error")
            return redirect(url_for("login"))

        return view(*args, **kwargs)

    return wrapped_view


# ==========================================
# 2. DATABASE TABLES
# ==========================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
class Trip(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
    db.Integer,
    db.ForeignKey("user.id"),
    nullable=True
)
    origin = db.Column(
        db.String(100), nullable=False, default="Not specified"
    )
    destination = db.Column(db.String(100), nullable=False)
    travellers = db.Column(db.Integer, nullable=False)
    days = db.Column(db.Integer, nullable=False)
    travel_mode = db.Column(db.String(30), nullable=False)
    hotel_category = db.Column(db.String(30), nullable=False)
    food_preference = db.Column(db.String(30), nullable=False)
    season = db.Column(db.String(30), nullable=False)
    predicted_cost = db.Column(db.Float, nullable=False)
    budget = db.Column(db.Float, nullable=True, default=0)
    created_at = db.Column(
        db.DateTime, server_default=db.func.now()
    )

    @property
    def people(self):
        return self.travellers


class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=True
    )
    destination = db.Column(db.String(100), nullable=False)
    created_at = db.Column(
        db.DateTime, server_default=db.func.now()
    )


class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    trip_id = db.Column(
        db.Integer, db.ForeignKey("trip.id"), nullable=False
    )
    category = db.Column(db.String(40), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    note = db.Column(db.String(250), default="")
    created_at = db.Column(
        db.DateTime, server_default=db.func.now()
    )

    @property
    def description(self):
        return self.note


class ItineraryItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    trip_id = db.Column(
        db.Integer, db.ForeignKey("trip.id"), nullable=False
    )
    day_number = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(120), nullable=False)
    notes = db.Column(db.String(500), default="")
    time = db.Column(db.String(20), default="")
    created_at = db.Column(
        db.DateTime, server_default=db.func.now()
    )

    @property
    def activity(self):
        return self.title


# ==========================================
# 3. DATABASE INITIALIZATION
# ==========================================

with app.app_context():
    db.create_all()

    with db.engine.begin() as connection:

        # TRIP TABLE
        trip_columns = [
            row[1] for row in connection.execute(
                text("PRAGMA table_info(trip)")
            )
        ]

        if "budget" not in trip_columns:
            connection.execute(
                text("ALTER TABLE trip ADD COLUMN budget FLOAT DEFAULT 0")
            )

        if "user_id" not in trip_columns:
            connection.execute(
                text("ALTER TABLE trip ADD COLUMN user_id INTEGER")
            )

        # FAVORITE TABLE
        favorite_columns = [
            row[1] for row in connection.execute(
                text("PRAGMA table_info(favorite)")
            )
        ]

        if "user_id" not in favorite_columns:
            connection.execute(
                text("ALTER TABLE favorite ADD COLUMN user_id INTEGER")
            )
            

        # ITINERARY TABLE
        itinerary_columns = [
            row[1] for row in connection.execute(
                text("PRAGMA table_info(itinerary_item)")
            )
        ]

        if "time" not in itinerary_columns:
            connection.execute(
                text("ALTER TABLE itinerary_item ADD COLUMN time VARCHAR(20) DEFAULT ''")
            )

# ==========================================
# 4. MACHINE LEARNING MODEL LOADING
# ==========================================

MODEL_DIR = BASE_DIR / "ml_model"
DATA_DIR = BASE_DIR / "data"

with open(MODEL_DIR / "model_metrics.json", "r", encoding="utf-8") as file:
    MODEL_METRICS = json.load(file)

cost_model = joblib.load(
    MODEL_DIR / "travel_cost_random_forest.pkl"
)

knn_bundle = joblib.load(
    MODEL_DIR / "destination_knn.pkl"
)

catalog = knn_bundle["catalog"]
knn = knn_bundle["model"]
knn_scaler = knn_bundle["scaler"]
knn_features = knn_bundle["features"]


# ==========================================
# 5. APPLICATION OPTIONS
# ==========================================

TRAVEL_MODES = [
    "Bus",
    "Train",
    "Car",
    "Flight"
]

HOTEL_CATEGORIES = [
    "Budget",
    "Standard",
    "Luxury"
]

FOOD_PREFERENCES = [
    "Budget",
    "Standard",
    "Premium"
]

SEASONS = [
    "Summer",
    "Winter",
    "Monsoon",
    "Peak Season",
    "Off Season"
]


# ==========================================
# 6. HELPER FUNCTIONS
# ==========================================


def make_cost_input(
    travellers,
    days,
    travel_mode,
    hotel_category,
    food_preference,
    season,
    destination
):
    destination_info = get_destination(destination)

    if destination_info is None:
        raise ValueError("Destination details not found.")

    return pd.DataFrame([{
        "destination": destination,
        "destination_type": destination_info["destination_type"],
        "distance_km": destination_info["distance_km"],
        "travellers": int(travellers),
        "days": int(days),
        "travel_mode": travel_mode,
        "hotel_category": hotel_category,
        "food_preference": food_preference,
        "season": season
    }])



def get_destination(destination_name):
    if "destination" not in catalog.columns:
        return None

    result = catalog[
        catalog["destination"].astype(str).str.lower()
        == str(destination_name).lower()
    ]

    if result.empty:
        return None

    return result.iloc[0].to_dict()


def format_currency(amount):
    return "₹{:,.0f}".format(float(amount))


def get_destination_list():
    if "destination" not in catalog.columns:
        return []

    return sorted(
        catalog["destination"].dropna().astype(str).unique().tolist()
    )


def get_trip_or_none(trip_id):
    return db.session.get(Trip, trip_id)


# ==========================================
# 7. COMMON TEMPLATE DATA
# ==========================================

@app.context_processor
def inject_common_data():
    return {
        "travel_modes": TRAVEL_MODES,
        "hotel_categories": HOTEL_CATEGORIES,
        "food_preferences": FOOD_PREFERENCES,
        "seasons": SEASONS,
        "destinations_list": get_destination_list()
    }
# ==========================================
# SIGN UP
# ==========================================

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("Please fill all required fields.", "error")
            return redirect(url_for("signup"))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("signup"))

        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
            return redirect(url_for("signup"))

        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            flash("Email already registered. Please login.", "error")
            return redirect(url_for("login"))

        new_user = User(
            name=name,
            email=email,
            password=generate_password_hash(password)
        )

        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")
# ==========================================
# LOGIN
# ==========================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):

            session["user_id"] = user.id
            session["user_name"] = user.name

            flash("Login successful!", "success")
            return redirect(url_for("index"))

        flash("Invalid email or password.", "error")
        return redirect(url_for("login"))

    return render_template("login.html")
# ==========================================
# LOGOUT
# ==========================================

@app.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out successfully.", "success")

    return redirect(url_for("login"))
# ==========================================
# 5. OVERVIEW
# ==========================================

@app.route("/")
def index():

    user_id = session["user_id"]

    # Recent trips of logged-in user
    recent = Trip.query.filter_by(
        user_id=user_id
    ).order_by(
        Trip.id.desc()
    ).limit(5).all()

    # User-specific totals
    total_trips = Trip.query.filter_by(
        user_id=user_id
    ).count()

    total_favorites = Favorite.query.filter_by(
        user_id=user_id
    ).count()

    total_cost = db.session.query(
        db.func.coalesce(
            db.func.sum(Trip.predicted_cost), 0
        )
    ).filter(
        Trip.user_id == user_id
    ).scalar()

    return render_template(
        "index.html",
        recent=recent,
        metrics=MODEL_METRICS,
        total_trips=total_trips,
        total_favorites=total_favorites,
        total_spent_estimate=float(total_cost or 0)
    )



# ==========================================
# 9. ABOUT PAGE
# ==========================================

@app.route("/about")
def about():
    return render_template(
        "about.html",
        model_metrics=MODEL_METRICS
    )


# ==========================================
# 10. PREDICTION PAGE
# ==========================================

@app.route("/predict", methods=["GET"], endpoint="predict_page")
def predict_page():

    destination_data = catalog.to_dict(orient="records")

    return render_template(
        "predict.html",
        destinations=destination_data,
        travel_modes=TRAVEL_MODES,
        hotel_categories=HOTEL_CATEGORIES,
        food_preferences=FOOD_PREFERENCES,
        seasons=SEASONS,
        selected_destination=request.args.get("destination", ""),
        selected_days=request.args.get("days", 1),
        selected_travellers=request.args.get("travellers", 1),
        selected_budget=request.args.get("budget", "")
    )



# ==========================================
# 11. AI TRAVEL COST PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"], endpoint="predict")
def predict():
    try:
        origin = request.form.get(
            "origin", "Not specified"
        )

        destination = request.form.get(
            "destination", ""
        )

        travellers = int(
            request.form.get("travellers", 1)
        )

        days = int(
            request.form.get("days", 1)
        )

        travel_mode = request.form.get(
            "travel_mode", "Bus"
        )

        hotel_category = request.form.get(
            "hotel_category", "Budget"
        )

        food_preference = request.form.get(
            "food_preference", "Budget"
        )

        season = request.form.get(
            "season", "Off Season"
        )
        budget_value = request.form.get("budget", "").strip()

        budget = float(budget_value) if budget_value else None

        if budget is not None and budget <= 0:
            flash("Please enter a valid budget.", "error")
            return redirect(url_for("predict_page"))

        if not destination:
            flash("Please select a destination.", "error")
            return redirect(url_for("predict_page"))

        if travellers < 1 or travellers > 20:
            flash("Travellers must be between 1 and 20.", "error")
            return redirect(url_for("predict_page"))

        if days < 1 or days > 60:
            flash("Trip duration must be between 1 and 60 days.", "error")
            return redirect(url_for("predict_page"))

        input_data = make_cost_input(
            travellers,
            days,
            travel_mode,
            hotel_category,
            food_preference,
            season,
            destination
        )

        predicted_cost = float(
            cost_model.predict(input_data)[0]
        )

        predicted_cost = max(0, predicted_cost)

        new_trip = Trip(
            user_id=session["user_id"],
            origin=origin,
            destination=destination,
            travellers=travellers,
            days=days,
            travel_mode=travel_mode,
            hotel_category=hotel_category,
            food_preference=food_preference,
            season=season,
            predicted_cost=predicted_cost,
            budget=budget
        )

        db.session.add(new_trip)
        db.session.commit()

        return render_template(
            "prediction_result.html",
            trip=new_trip,
            predicted_cost=predicted_cost,
            destination=destination,
            origin=origin
        )

    except Exception as error:
        db.session.rollback()
        print("Prediction Error:", error)
        flash(
            "Prediction could not be completed. Please check your input.",
            "error"
        )
        return redirect(url_for("predict_page"))


# ==========================================
# 12. TRIP HISTORY
# ==========================================


@app.route("/history")
def history():
    trips = Trip.query.filter_by(
        user_id=session["user_id"]
    ).order_by(
        Trip.created_at.desc()
    ).all()

    return render_template(
        "history.html",
        trips=trips,
        history=trips
    )

# ==========================================
# 13. DESTINATIONS PAGE
# ==========================================

@app.route("/destinations")
def destinations():
    destination_data = catalog.to_dict(
        orient="records"
    )

    return render_template(
        "destinations.html",
        destinations=destination_data
    )


# ==========================================
# 14. DESTINATION DETAILS
# ==========================================

@app.route("/destination/<path:destination_name>")
def destination_detail(destination_name):
    destination = get_destination(destination_name)

    if destination is None:
        flash("Destination not found.", "error")
        return redirect(url_for("destinations"))

    return render_template(
        "destination_detail.html",
        destination=destination
    )


# ==========================================
# 15. PLANNER PAGE
# ==========================================


@app.route("/planner")
def planner():
    destination_data = catalog.to_dict(orient="records")

    return render_template(
        "planner.html",
        destinations=destination_data
    )


# ==========================================
# 16. DESTINATION RECOMMENDATIONS (KNN)
# ==========================================

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    recommendations = []
    budget = 0
    days = 1
    people = 1

    if request.method == "POST":
        try:
            budget = float(request.form.get("budget", 0))
            days = int(request.form.get("days", 1))
            people = int(request.form.get("people", 1))

            if budget <= 0 or days <= 0 or people <= 0:
                flash("Enter valid budget, days and people.", "error")
            else:
                user_values = {}

                for feature in knn_features:
                    name = str(feature).lower()

                    if "budget" in name or "cost" in name or "price" in name:
                        user_values[feature] = budget
                    elif "day" in name:
                        user_values[feature] = days
                    elif "people" in name or "traveller" in name or "person" in name:
                        user_values[feature] = people
                    else:
                        user_values[feature] = 0

                input_df = pd.DataFrame(
                    [user_values],
                    columns=knn_features
                )

                scaled_input = knn_scaler.transform(input_df)

                distances, indices = knn.kneighbors(
                    scaled_input,
                    n_neighbors=min(5, len(catalog))
                )

                for index in indices[0]:
                    item = catalog.iloc[int(index)].to_dict()
                    recommendations.append(item)

        except Exception as error:
            print("Recommendation Error:", error)
            flash("Unable to generate recommendations.", "error")

    return render_template(
        "recommend.html",
        recommendations=recommendations,
        budget=budget,
        days=days,
        people=people
    )


# ==========================================
# 17. FAVORITES
# ==========================================


@app.route("/favorites")
def favorites():
    favorite_items = Favorite.query.filter_by(
        user_id=session["user_id"]
    ).order_by(
        Favorite.created_at.desc()
    ).all()

    return render_template(
        "favorites.html",
        favorites=favorite_items
    )


# ==========================================
# 18. ADD / REMOVE FAVORITE
# ==========================================

@app.route(
    "/favorites/toggle/<path:destination_name>",
    methods=["GET", "POST"]
)
def toggle_favorite(destination_name):
    existing = Favorite.query.filter_by(
        destination=destination_name
    ).first()

    if existing:
        db.session.delete(existing)
        flash("Destination removed from favorites.", "success")
    else:
        new_favorite = Favorite(
            user_id=session["user_id"],
            destination=destination_name
        )
        db.session.add(new_favorite)
        flash("Destination added to favorites.", "success")

    db.session.commit()

    return redirect(
        request.referrer or url_for("favorites")
    )


# ==========================================
# TRIP DETAILS
# ==========================================

@app.route("/trip/<int:trip_id>")
def trip_detail(trip_id):

    trip = Trip.query.filter_by(
        id=trip_id,
        user_id=session["user_id"]
    ).first_or_404()

    expenses_list = Expense.query.filter_by(
        trip_id=trip.id
    ).order_by(
        Expense.id.desc()
    ).all()

    itinerary_list = ItineraryItem.query.filter_by(
        trip_id=trip.id
    ).order_by(
        ItineraryItem.day_number,
        ItineraryItem.id
    ).all()

    actual_total = sum(
        item.amount for item in expenses_list
    )

    return render_template(
        "trip_detail.html",
        trip=trip,
        expenses=expenses_list,
        itinerary=itinerary_list,
        actual_total=actual_total
    )


# ==========================================
# DELETE TRIP
# ==========================================

@app.route("/trip/<int:trip_id>/delete", methods=["POST"])
def delete_trip(trip_id):

    trip = Trip.query.filter_by(
        id=trip_id,
        user_id=session["user_id"]
    ).first_or_404()

    Expense.query.filter_by(
        trip_id=trip.id
    ).delete()

    ItineraryItem.query.filter_by(
        trip_id=trip.id
    ).delete()

    db.session.delete(trip)
    db.session.commit()

    flash("Trip deleted successfully.", "success")

    return redirect(url_for("history"))

# ==========================================
# 21. EXPENSE MANAGEMENT
# ==========================================

@app.route("/expenses", methods=["GET", "POST"])
def expenses():

    user_id = session["user_id"]

    # Show only logged-in user's trips
    trips = Trip.query.filter_by(
        user_id=user_id
    ).order_by(
        Trip.created_at.desc()
    ).all()

    if request.method == "POST":
        try:
            trip_id = int(request.form.get("trip_id"))
            category = request.form.get("category", "Other")
            amount = float(request.form.get("amount", 0))
            note = request.form.get("note", "")

            # Check whether this trip belongs to the logged-in user
            trip = Trip.query.filter_by(
                id=trip_id,
                user_id=user_id
            ).first()

            if trip is None or amount <= 0:
                flash("Please enter valid expense details.", "error")
            else:
                new_expense = Expense(
                    trip_id=trip.id,
                    category=category,
                    amount=amount,
                    note=note
                )

                db.session.add(new_expense)
                db.session.commit()

                flash("Expense added successfully.", "success")

        except Exception as error:
            db.session.rollback()
            print("Expense Error:", error)
            flash("Unable to add expense.", "error")

        return redirect(url_for("expenses"))

    # Show expenses only for this user's trips
    expense_items = Expense.query.join(
        Trip,
        Expense.trip_id == Trip.id
    ).filter(
        Trip.user_id == user_id
    ).order_by(
        Expense.created_at.desc()
    ).all()

    total_expenses = sum(
        item.amount for item in expense_items
    )

    return render_template(
        "expenses.html",
        trips=trips,
        expenses=expense_items,
        total_expenses=total_expenses
    )


# ==========================================
# 22. DELETE EXPENSE
# ==========================================

@app.route(
    "/expense/<int:expense_id>/delete",
    methods=["POST", "GET"]
)
def delete_expense(expense_id):

    expense = Expense.query.join(
        Trip,
        Expense.trip_id == Trip.id
    ).filter(
        Expense.id == expense_id,
        Trip.user_id == session["user_id"]
    ).first()

    if expense is None:
        flash("Expense not found.", "error")
    else:
        db.session.delete(expense)
        db.session.commit()
        flash("Expense deleted successfully.", "success")

    return redirect(url_for("expenses"))


# ==========================================
# 23. ITINERARY MANAGEMENT
# ==========================================

@app.route("/itinerary", methods=["GET", "POST"])
def itinerary():

    user_id = session["user_id"]

    # Show only logged-in user's trips
    trips = Trip.query.filter_by(
        user_id=user_id
    ).order_by(
        Trip.created_at.desc()
    ).all()

    if request.method == "POST":
        try:
            trip_id = int(request.form.get("trip_id"))
            day_number = int(request.form.get("day_number"))
            title = request.form.get("title", "").strip()
            notes = request.form.get("notes", "")
            time = request.form.get("time", "")

            # Verify trip ownership
            trip = Trip.query.filter_by(
                id=trip_id,
                user_id=user_id
            ).first()

            if trip is None or not title:
                flash("Please enter valid itinerary details.", "error")

            elif day_number < 1 or day_number > trip.days:
                flash("Day number must be within the trip duration.", "error")

            else:
                new_item = ItineraryItem(
                    trip_id=trip.id,
                    day_number=day_number,
                    title=title,
                    notes=notes,
                    time=time
                )

                db.session.add(new_item)
                db.session.commit()

                flash("Itinerary item added successfully.", "success")

        except Exception as error:
            db.session.rollback()
            print("Itinerary Error:", error)
            flash("Unable to add itinerary item.", "error")

        return redirect(url_for("itinerary"))

    # Show itinerary items only from this user's trips
    items = ItineraryItem.query.join(
        Trip,
        ItineraryItem.trip_id == Trip.id
    ).filter(
        Trip.user_id == user_id
    ).order_by(
        ItineraryItem.day_number
    ).all()

    return render_template(
        "itinerary.html",
        trips=trips,
        items=items,
        itinerary=items
    )


# ==========================================
# 24. DELETE ITINERARY ITEM
# ==========================================

@app.route(
    "/itinerary/item/<int:item_id>/delete",
    methods=["POST", "GET"]
)
def delete_itinerary_item(item_id):

    item = ItineraryItem.query.join(
        Trip,
        ItineraryItem.trip_id == Trip.id
    ).filter(
        ItineraryItem.id == item_id,
        Trip.user_id == session["user_id"]
    ).first()

    if item is None:
        flash("Itinerary item not found.", "error")
    else:
        db.session.delete(item)
        db.session.commit()
        flash("Itinerary item deleted successfully.", "success")

    return redirect(url_for("itinerary"))

# ==========================================
# 25. REPORTS AND ANALYTICS
# ==========================================


@app.route("/reports")
def reports():
    trips = Trip.query.filter_by(
        user_id=session["user_id"]
    ).order_by(
        Trip.created_at.desc()
    ).all()

    total_trips = len(trips)

    total_predicted_cost = sum(
        float(trip.predicted_cost or 0)
        for trip in trips
    )

    user_trip_ids = [trip.id for trip in trips]

    total_expenses = 0

    if user_trip_ids:
        total_expenses = db.session.query(
            db.func.coalesce(db.func.sum(Expense.amount), 0)
        ).filter(
            Expense.trip_id.in_(user_trip_ids)
        ).scalar()

    average_trip_cost = (
        total_predicted_cost / total_trips
        if total_trips > 0 else 0
    )

    return render_template(
        "reports.html",
        trips=trips,
        total_trips=total_trips,
        total_predicted_cost=total_predicted_cost,
        total_expenses=total_expenses,
        average_trip_cost=average_trip_cost
    )


# ==========================================
# 26. SIMPLE AI TRAVEL ASSISTANT
# ==========================================
def get_travel_assistant_answer(question):
    question = question.lower()

    if any(word in question for word in ["budget", "cheap", "affordable", "money", "cost"]):
        return """
        You can travel within your budget by following these tips:

        1. Choose a destination that matches your budget.
        2. Prefer bus or train instead of flights.
        3. Choose budget hotels or affordable accommodation.
        4. Eat at local restaurants.
        5. Visit free or low-cost tourist attractions.
        6. Keep some money for unexpected expenses.

        Tell me your budget, destination and trip duration
        to get more specific travel suggestions.
        """

    elif "hotel" in question or "stay" in question:
        return "Choose budget hotels, compare room prices and book early to reduce accommodation expenses."

    elif "food" in question:
        return "Try local restaurants, set a daily food budget and avoid expensive tourist-area restaurants."

    elif "transport" in question or "travel" in question:
        return "Compare bus, train and flight costs. For short-distance trips, buses and trains can help reduce expenses."

    else:
        return "I can help with travel budgets, destinations, hotels, food and transportation. Ask me a specific travel question."

@app.route("/assistant", methods=["GET", "POST"])
def assistant():
    question = ""
    answer = ""

    if request.method == "POST":
        question = request.form.get(
            "question", ""
        ).strip()

        q = question.lower()

        if not question:
            answer = "Please enter your travel question."

        elif "budget" in q or "cost" in q or "price" in q:
            answer = (
                "To estimate your travel budget, open AI Prediction. "
                "Enter your destination, number of travellers, trip days "
                "and travel preferences. SmartTrip AI will estimate the cost."
            )

        elif "destination" in q or "place" in q:
            answer = (
                "Open Destination Recommendations and enter your budget, "
                "trip duration and number of travellers to explore suitable "
                "destinations using the KNN recommendation model."
            )

        elif "itinerary" in q or "plan" in q:
            answer = (
                "Open Itinerary to add your daily travel activities, "
                "time and notes for a saved trip."
            )

        elif "expense" in q:
            answer = (
                "Open Expense Management to record and track your trip expenses."
            )

        elif "history" in q or "previous trip" in q:
            answer = (
                "Open Prediction History to view your previously saved trips."
            )

        else:
            answer = (
                "I can help with travel budget prediction, destination "
                "recommendations, itinerary planning and expense tracking. "
                "Please ask about one of these topics."
            )
    answer = get_travel_assistant_answer(question)
    return render_template(
        "assistant.html",
        question=question,
        answer=answer
    )


# ==========================================
# 27. SETTINGS PAGE
# ==========================================

@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        flash(
            "Settings are currently available as a demonstration page.",
            "success"
        )
        return redirect(url_for("settings"))

    return render_template(
    "settings.html",
    total_trips=Trip.query.count(),
    total_favorites=Favorite.query.count(),
    total_expenses=Expense.query.count(),
    total_itinerary=ItineraryItem.query.count()
)


# ==========================================
# 28. HEALTH CHECK
# ==========================================

@app.route("/health")
def health():
    return jsonify({
        "status": "success",
        "application": "SmartTrip AI",
        "message": "Application is running"
    })

# ==========================================
# 29. APPLICATION START
# ==========================================

# ==========================================
# PROTECT ALL WEBSITE PAGES
# ==========================================

@app.before_request
def check_login():

    public_pages = ["login", "signup", "static"]

    if request.endpoint not in public_pages:
        if "user_id" not in session:
            return redirect(url_for("login"))
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )

