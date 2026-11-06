"""Constants and the loader for this project's fitness-app user file.

Kept apart from the generic template modules on purpose: everything here is
about this one dataset, so the next project deletes it rather than reusing it.
"""

import numpy as np
import pandas as pd

from src.paths import PROCESSED

RAW_FILE = 'dataset for assignment 2-0426.csv'
PROCESSED_FILE = 'users.csv'

# Raw header -> analysis name. The raw file spells it "Travelled" (NZ/UK);
# the brief's sample table spells it "Traveled".
COLUMNS = {
    'User ID': 'user_id',
    'Gender': 'gender',
    'Age': 'age',
    'Activity Level': 'activity_level',
    'Location': 'location',
    'App Sessions': 'app_sessions',
    'Distance Travelled (km)': 'distance_km',
    'Calories Burned': 'calories',
}

TIERS = ['Sedentary', 'Moderate', 'Active']          # ordered by app sessions
LOCATIONS = ['Urban', 'Suburban', 'Rural']
GENDERS = ['Female', 'Male']
AGE_EDGES = [17, 29, 39, 49, 59]
AGE_LABELS = ['18–29', '30–39', '40–49', '50–59']

# One hue, light to dark, because the tiers are ordered. Distinct in greyscale
# and under the common colour-vision deficiencies.
TIER_COLORS = {'Sedentary': '#a9cbe6', 'Moderate': '#4f8fc6', 'Active': '#1d4f86'}


# Notebook 01's findings, as data.
COPY_BLOCK_START = 5001                      # IDs 5001-5299: 272 verbatim + 27 altered copies
SUSPECT_DISTANCES = {1999: 92.0, 2067: 100.0}  # x10 slips: user -> value with the digit reversed


def clean_users(raw, drop_copy_block=True, suspect_distances='blank'):
    """Apply notebook 01's decisions to the raw file and add the derived fields.

    The defaults ARE the decisions. The alternatives exist so the sensitivity
    analysis in 04 can show what each choice changes:

    - drop_copy_block: remove IDs 5001-5299 (copies of users 3580-3855).
    - suspect_distances: 'blank' sets the two x10 distances to missing,
      'keep' leaves them as recorded, 'correct' reverses the slipped digit.
    """
    if suspect_distances not in ('blank', 'keep', 'correct'):
        raise ValueError("suspect_distances must be 'blank', 'keep' or 'correct'")
    df = raw.rename(columns=COLUMNS)
    if drop_copy_block:
        df = df[df['user_id'] < COPY_BLOCK_START]
    df = df.copy()

    suspect = df['user_id'].isin(SUSPECT_DISTANCES)
    if suspect_distances == 'blank':
        df.loc[suspect, 'distance_km'] = np.nan
    elif suspect_distances == 'correct':
        df.loc[suspect, 'distance_km'] = df.loc[suspect, 'user_id'].map(SUSPECT_DISTANCES)
    df['distance_flag'] = np.select(
        [suspect, df['distance_km'].isna()],
        [f'suspected x10 entry error ({suspect_distances})', 'missing'], default='')

    df['km_per_session'] = df['distance_km'] / df['app_sessions']
    df['kcal_per_session'] = df['calories'] / df['app_sessions']
    df['age_band'] = add_age_band(df)
    return df.reset_index(drop=True)


def add_age_band(df, col='age'):
    """Ten-year age bands (18–29 is twelve years: the data start at 18)."""
    return pd.cut(df[col], AGE_EDGES, labels=AGE_LABELS, ordered=True)


def load_users():
    """The cleaned user table written by notebook 02, with categories ordered."""
    df = pd.read_csv(PROCESSED / PROCESSED_FILE)
    df['activity_level'] = pd.Categorical(df['activity_level'], TIERS, ordered=True)
    df['location'] = pd.Categorical(df['location'], LOCATIONS)
    df['gender'] = pd.Categorical(df['gender'], GENDERS)
    df['age_band'] = pd.Categorical(df['age_band'], AGE_LABELS, ordered=True)
    return df


# Inference contract for the tier model (notebook 04). The model reads only
# distance and calories, so the checks are on those two numbers: present,
# positive, and consistent with the per-session invariants found in 02
# (0.5-1.0 km and 2-5 kcal per session, sessions 50-199).
KM_PER_SESSION = (0.5, 1.0)
KCAL_PER_SESSION = (2.0, 5.0)
SESSION_RANGE = (50, 199)


def check_tier_inputs(distance_km, calories):
    """Why each imported (distance, calories) pair falls outside what the tier
    model was built on; an empty string means it passes.

    A pair passes when some whole number of sessions in SESSION_RANGE could have
    produced both values: distance / 1.0 <= sessions <= distance / 0.5 and
    calories / 5 <= sessions <= calories / 2 (half a unit of rounding allowed).
    A record that fails should not get a model prediction: the app falls back to
    asking the user, or to the default starting plan.
    """
    d = pd.to_numeric(pd.Series(distance_km), errors='coerce').to_numpy(float)
    c = pd.to_numeric(pd.Series(calories), errors='coerce').to_numpy(float)
    lo = np.maximum.reduce([(d - 0.5) / KM_PER_SESSION[1], (c - 0.5) / KCAL_PER_SESSION[1],
                            np.full_like(d, SESSION_RANGE[0])])
    hi = np.minimum.reduce([(d + 0.5) / KM_PER_SESSION[0], (c + 0.5) / KCAL_PER_SESSION[0],
                            np.full_like(d, SESSION_RANGE[1])])
    reasons = np.where(np.isnan(d) | np.isnan(c), 'missing value',
              np.where((d <= 0) | (c <= 0), 'not positive',
              np.where(np.ceil(lo) > np.floor(hi), 'no session count fits both values', '')))
    return pd.Series(reasons, index=distance_km.index if isinstance(distance_km, pd.Series) else None, name='contract_violation')
