import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output folder
os.makedirs("EDA_graphs", exist_ok=True)


TEST_PATH = "test_set_VU_DM.csv"
TRAIN_PATH = "training_set_VU_DM.csv"

print("Loading data...")
dfTrain = pd.read_csv(TRAIN_PATH, low_memory=False)
dfTest = pd.read_csv(TEST_PATH, low_memory=False)
print(f"  Train: {dfTrain.shape[0]:,} rows, {dfTrain.shape[1]} columns")
print(f"  Test:  {dfTest.shape[0]:,} rows, {dfTest.shape[1]} columns")


# BASIC STATS
print(f"\nColumns: {dfTrain.columns.tolist()}")
print(dfTrain.head())
print(dfTrain.dtypes)

print("\nMissingness values")
missing_pct = (dfTrain.isna().mean() * 100).sort_values(ascending=False)
print(missing_pct[missing_pct > 0].round(2))

print(dfTrain.describe())

print("\nSample unique counts:")
print(dfTrain.nunique().sort_values(ascending=False).head(20))


# DISTRIBUTION SHIFT CHECK
for col in ['price_usd', 'prop_starrating', 'prop_review_score',
            'srch_booking_window', 'orig_destination_distance']:
    print(f"\n{col}")
    print(f"  Train mean: {dfTrain[col].mean():.3f}, null: {dfTrain[col].isna().mean():.3f}")
    print(f"  Test mean:  {dfTest[col].mean():.3f}, null: {dfTest[col].isna().mean():.3f}")


# TARGET VARIABLE STATS
print(f"\nBooking rate: {dfTrain['booking_bool'].mean():.4f}")
print(f"Click rate: {dfTrain['click_bool'].mean():.4f}")
print(f"\nHotels per search:")
print(dfTrain.groupby('srch_id').size().describe())
print(f"\nBookings per search:")
print(dfTrain.groupby('srch_id')['booking_bool'].sum().value_counts().sort_index())

print("\nPrice by booking status:")
print(dfTrain.groupby('booking_bool')['price_usd'].describe())

print("\nBooking rate by star rating:")
print(dfTrain.groupby('prop_starrating')['booking_bool'].mean())

print("\nBooking rate by review score:")
print(dfTrain.groupby('prop_review_score')['booking_bool'].mean())

dfTrain['has_history'] = dfTrain['visitor_hist_starrating'].notna()
print("\nBooking rate by visitor history:")
print(dfTrain.groupby('has_history')['booking_bool'].mean())

for i in range(1, 9):
    col = f'comp{i}_rate'
    if col in dfTrain.columns:
        pct_null = dfTrain[col].isna().mean()
        print(f"{col}: {pct_null:.1%} null")

numeric_cols = dfTrain.select_dtypes(include='number').columns
corrs = dfTrain[numeric_cols].corrwith(dfTrain['booking_bool']).sort_values(key=abs, ascending=False)
print("\nCorrelation with booking_bool:")
print(corrs.head(20))


# ═══════════════════════════════════════════════════════════════════════════
# GRAPHS
# ═══════════════════════════════════════════════════════════════════════════
print("\n\nGenerating graphs...")


# POSITION BIAS
fig, ax = plt.subplots(figsize=(10, 5))
pos_rates = dfTrain.groupby('position')[['click_bool', 'booking_bool']].mean().head(20)
ax.plot(pos_rates.index, pos_rates['click_bool'], marker='o', color='#3498db', label='Click Rate')
ax.plot(pos_rates.index, pos_rates['booking_bool'], marker='o', color='#e74c3c', label='Booking Rate')
ax.set_title('Click and Booking Rate by Position')
ax.set_xlabel('Position on Results Page')
ax.set_ylabel('Rate')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('EDA_graphs/01_position_bias.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 01_position_bias.png")


# POSITION BIAS: RANDOM VS NORMAL SORT
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, label in enumerate([0, 1]):
    subset = dfTrain[dfTrain['random_bool'] == label]
    rates = subset.groupby('position')[['click_bool', 'booking_bool']].mean().head(20)
    axes[idx].plot(rates.index, rates['click_bool'], marker='o', color='#3498db', label='Click Rate')
    axes[idx].plot(rates.index, rates['booking_bool'], marker='o', color='#e74c3c', label='Booking Rate')
    title = 'Normal Sort Order' if label == 0 else 'Random Sort Order'
    axes[idx].set_title(f'{title} (random_bool={label})')
    axes[idx].set_xlabel('Position')
    axes[idx].set_ylabel('Rate')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)
plt.suptitle('Position Bias: Normal vs Random Sort', fontsize=14)
plt.tight_layout()
plt.savefig('EDA_graphs/02_position_bias_random_vs_normal.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 02_position_bias_random_vs_normal.png")


# MISSINGNESS BAR CHART
fig, ax = plt.subplots(figsize=(12, 6))
missing = missing_pct[missing_pct > 0].sort_values(ascending=True)
colors = ['#e74c3c' if v > 50 else '#7f8c8d' if v > 20 else '#3498db' for v in missing.values]
missing.plot(kind='barh', ax=ax, color=colors)
ax.set_title('Missing Values by Feature')
ax.set_xlabel('Missing (%)')
ax.axvline(x=50, color='#e74c3c', linestyle='--', alpha=0.5, label='50% threshold')
ax.legend()
plt.savefig('EDA_graphs/03_missingness.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 03_missingness.png")


# PRICE DISTRIBUTION: BOOKED VS NOT BOOKED
fig, ax = plt.subplots(figsize=(10, 5))
booked = dfTrain[dfTrain['booking_bool'] == 1]['price_usd'].clip(upper=1000)
not_booked = dfTrain[dfTrain['booking_bool'] == 0]['price_usd'].clip(upper=1000)
ax.hist(not_booked, bins=50, alpha=0.5, label='Not Booked', density=True, color='#3498db')
ax.hist(booked, bins=50, alpha=0.5, label='Booked', density=True, color='#e74c3c')
ax.set_title('Price Distribution: Booked vs Not Booked')
ax.set_xlabel('Price (USD, clipped at $1000)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('EDA_graphs/04_price_distribution_by_booking.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 04_price_distribution_by_booking.png")


# BOOKING RATE BY STAR RATING
fig, ax = plt.subplots(figsize=(8, 5))
star_rates = dfTrain.groupby('prop_starrating')['booking_bool'].agg(['mean', 'count'])
bars = ax.bar(star_rates.index, star_rates['mean'], color='#3498db', edgecolor='black')
ax.set_title('Booking Rate by Star Rating')
ax.set_xlabel('Star Rating')
ax.set_ylabel('Booking Rate')
ax.set_xticks(range(0, 6))
ax.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, star_rates['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'n={count:,}', ha='center', va='bottom', fontsize=8)
plt.savefig('EDA_graphs/05_booking_rate_by_stars.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 05_booking_rate_by_stars.png")


# BOOKING RATE BY REVIEW SCORE
fig, ax = plt.subplots(figsize=(10, 5))
review_rates = dfTrain.groupby('prop_review_score')['booking_bool'].agg(['mean', 'count'])
ax.bar(review_rates.index.astype(str), review_rates['mean'], color='#3498db', edgecolor='black')
ax.set_title('Booking Rate by Review Score')
ax.set_xlabel('Review Score')
ax.set_ylabel('Booking Rate')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45)
plt.savefig('EDA_graphs/06_booking_rate_by_review.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 06_booking_rate_by_review.png")


# CORRELATION WITH BOOKING (TOP 20)
fig, ax = plt.subplots(figsize=(10, 7))
top_corrs = corrs.drop(['booking_bool', 'click_bool']).head(18)
colors = ['#3498db' if v > 0 else '#e74c3c' for v in top_corrs.values]
top_corrs.plot(kind='barh', ax=ax, color=colors)
ax.set_title('Top Feature Correlations with Booking')
ax.set_xlabel('Correlation Coefficient')
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')
plt.savefig('EDA_graphs/07_correlation_with_booking.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 07_correlation_with_booking.png")


# HOTELS PER SEARCH DISTRIBUTION
fig, ax = plt.subplots(figsize=(10, 5))
hotels_per_search = dfTrain.groupby('srch_id').size()
ax.hist(hotels_per_search, bins=range(0, 42), edgecolor='black', color='#3498db', alpha=0.7)
ax.set_title('Distribution of Hotels per Search')
ax.set_xlabel('Number of Hotels in Search Results')
ax.set_ylabel('Number of Searches')
ax.axvline(x=hotels_per_search.mean(), color='#e74c3c', linestyle='--', label=f'Mean: {hotels_per_search.mean():.1f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.savefig('EDA_graphs/08_hotels_per_search.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 08_hotels_per_search.png")


# BOOKING RATE BY PROMOTION FLAG
fig, ax = plt.subplots(figsize=(6, 5))
promo_rates = dfTrain.groupby('promotion_flag')['booking_bool'].agg(['mean', 'count'])
bars = ax.bar(['No Promotion', 'Promotion'], promo_rates['mean'], color=['#7f8c8d', '#3498db'], edgecolor='black')
ax.set_title('Booking Rate: Promotion vs No Promotion')
ax.set_ylabel('Booking Rate')
ax.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, promo_rates['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'n={count:,}', ha='center', va='bottom', fontsize=9)
plt.savefig('EDA_graphs/09_booking_rate_by_promotion.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 09_booking_rate_by_promotion.png")


# BOOKING RATE BY BRAND
fig, ax = plt.subplots(figsize=(6, 5))
brand_rates = dfTrain.groupby('prop_brand_bool')['booking_bool'].agg(['mean', 'count'])
bars = ax.bar(['Independent', 'Major Chain'], brand_rates['mean'], color=['#7f8c8d', '#3498db'], edgecolor='black')
ax.set_title('Booking Rate: Independent vs Major Chain Hotels')
ax.set_ylabel('Booking Rate')
ax.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, brand_rates['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'n={count:,}', ha='center', va='bottom', fontsize=9)
plt.savefig('EDA_graphs/10_booking_rate_by_brand.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 10_booking_rate_by_brand.png")


# VISITOR HISTORY: BOOKING RATE WITH VS WITHOUT
fig, ax = plt.subplots(figsize=(6, 5))
hist_rates = dfTrain.groupby('has_history')['booking_bool'].agg(['mean', 'count'])
bars = ax.bar(['No History (95%)', 'Has History (5%)'], hist_rates['mean'],
              color=['#7f8c8d', '#3498db'], edgecolor='black')
ax.set_title('Booking Rate by Visitor Purchase History')
ax.set_ylabel('Booking Rate')
ax.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, hist_rates['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'n={count:,}', ha='center', va='bottom', fontsize=9)
plt.savefig('EDA_graphs/11_booking_rate_by_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 11_booking_rate_by_history.png")


# LOCATION SCORE 2 VS BOOKING
fig, ax = plt.subplots(figsize=(10, 5))
loc2 = dfTrain['prop_location_score2'].dropna()
loc2_booked = dfTrain.loc[dfTrain['booking_bool'] == 1, 'prop_location_score2'].dropna()
ax.hist(loc2, bins=50, alpha=0.5, label='All Hotels', density=True, color='#3498db')
ax.hist(loc2_booked, bins=50, alpha=0.5, label='Booked Hotels', density=True, color='#e74c3c')
ax.set_title('Location Score 2 Distribution: All vs Booked Hotels')
ax.set_xlabel('prop_location_score2')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('EDA_graphs/12_location_score2_by_booking.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 12_location_score2_by_booking.png")


# BOOKING WINDOW DISTRIBUTION
fig, ax = plt.subplots(figsize=(10, 5))
bw_booked = dfTrain.loc[dfTrain['booking_bool'] == 1, 'srch_booking_window'].clip(upper=200)
bw_all = dfTrain['srch_booking_window'].clip(upper=200)
ax.hist(bw_all, bins=50, alpha=0.5, label='All Searches', density=True, color='#3498db')
ax.hist(bw_booked, bins=50, alpha=0.5, label='Booked', density=True, color='#e74c3c')
ax.set_title('Booking Window Distribution: All vs Booked')
ax.set_xlabel('Days Until Check-in (clipped at 200)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('EDA_graphs/13_booking_window_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 13_booking_window_distribution.png")


# SATURDAY NIGHT STAY VS BOOKING
fig, ax = plt.subplots(figsize=(6, 5))
sat_rates = dfTrain.groupby('srch_saturday_night_bool')['booking_bool'].agg(['mean', 'count'])
bars = ax.bar(['No Saturday Night', 'Saturday Night'], sat_rates['mean'],
              color=['#7f8c8d', '#3498db'], edgecolor='black')
ax.set_title('Booking Rate: Weekend vs Non-Weekend Stays')
ax.set_ylabel('Booking Rate')
ax.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, sat_rates['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'n={count:,}', ha='center', va='bottom', fontsize=9)
plt.savefig('EDA_graphs/14_booking_rate_saturday_night.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 14_booking_rate_saturday_night.png")


# LENGTH OF STAY VS BOOKING RATE
fig, ax = plt.subplots(figsize=(10, 5))
los_rates = dfTrain.groupby('srch_length_of_stay')['booking_bool'].agg(['mean', 'count'])
los_rates = los_rates[los_rates['count'] > 1000]
ax.bar(los_rates.index, los_rates['mean'], color='#3498db', edgecolor='black')
ax.set_title('Booking Rate by Length of Stay')
ax.set_xlabel('Number of Nights')
ax.set_ylabel('Booking Rate')
ax.grid(True, alpha=0.3, axis='y')
plt.savefig('EDA_graphs/15_booking_rate_by_length_of_stay.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 15_booking_rate_by_length_of_stay.png")


# COMPETITOR RATE ADVANTAGE VS BOOKING
fig, ax = plt.subplots(figsize=(10, 5))
comp_booking = []
for i in range(1, 9):
    col = f'comp{i}_rate'
    if col in dfTrain.columns:
        cheaper = dfTrain.loc[dfTrain[col] == 1, 'booking_bool'].mean()
        same = dfTrain.loc[dfTrain[col] == 0, 'booking_bool'].mean()
        expensive = dfTrain.loc[dfTrain[col] == -1, 'booking_bool'].mean()
        comp_booking.append({'competitor': f'Comp {i}', 'Expedia Cheaper': cheaper,
                            'Same Price': same, 'Expedia More Expensive': expensive})
comp_df = pd.DataFrame(comp_booking).set_index('competitor')
comp_df.plot(kind='bar', ax=ax, color=['#3498db', '#7f8c8d', '#e74c3c'], edgecolor='black')
ax.set_title('Booking Rate by Expedia Price Competitiveness')
ax.set_ylabel('Booking Rate')
ax.set_xlabel('')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=0)
plt.legend(loc='upper right')
plt.savefig('EDA_graphs/16_competitor_rate_vs_booking.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 16_competitor_rate_vs_booking.png")


# TRAIN VS TEST DISTRIBUTION COMPARISON
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
compare_cols = ['price_usd', 'prop_starrating', 'prop_review_score',
                'srch_booking_window', 'srch_length_of_stay', 'orig_destination_distance']
for ax, col in zip(axes.flatten(), compare_cols):
    train_vals = dfTrain[col].dropna().clip(upper=dfTrain[col].quantile(0.99))
    test_vals = dfTest[col].dropna().clip(upper=dfTest[col].dropna().quantile(0.99))
    ax.hist(train_vals, bins=40, alpha=0.5, label='Train', density=True, color='#3498db')
    ax.hist(test_vals, bins=40, alpha=0.5, label='Test', density=True, color='#e74c3c')
    ax.set_title(f'{col}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle('Train vs Test Feature Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('EDA_graphs/17_train_vs_test_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 17_train_vs_test_distributions.png")


# ROOM COUNT AND GUEST COUNT VS BOOKING
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

room_rates = dfTrain.groupby('srch_room_count')['booking_bool'].mean()
room_rates = room_rates[room_rates.index <= 5]
axes[0].bar(room_rates.index, room_rates.values, color='#3498db', edgecolor='black')
axes[0].set_title('Booking Rate by Room Count')
axes[0].set_xlabel('Rooms')
axes[0].set_ylabel('Booking Rate')
axes[0].grid(True, alpha=0.3, axis='y')

adult_rates = dfTrain.groupby('srch_adults_count')['booking_bool'].mean()
adult_rates = adult_rates[adult_rates.index <= 6]
axes[1].bar(adult_rates.index, adult_rates.values, color='#3498db', edgecolor='black')
axes[1].set_title('Booking Rate by Adult Count')
axes[1].set_xlabel('Adults')
axes[1].grid(True, alpha=0.3, axis='y')

child_rates = dfTrain.groupby('srch_children_count')['booking_bool'].mean()
child_rates = child_rates[child_rates.index <= 5]
axes[2].bar(child_rates.index, child_rates.values, color='#3498db', edgecolor='black')
axes[2].set_title('Booking Rate by Children Count')
axes[2].set_xlabel('Children')
axes[2].grid(True, alpha=0.3, axis='y')

plt.suptitle('Booking Rate by Guest Configuration', fontsize=14)
plt.tight_layout()
plt.savefig('EDA_graphs/18_booking_rate_by_guests.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 18_booking_rate_by_guests.png")


# PRICE WITHIN SEARCH: BOOKED HOTEL RANK
fig, ax = plt.subplots(figsize=(10, 5))
dfTrain['price_rank'] = dfTrain.groupby('srch_id')['price_usd'].rank(method='min')
booked_ranks = dfTrain.loc[dfTrain['booking_bool'] == 1, 'price_rank']
ax.hist(booked_ranks, bins=range(1, 35), edgecolor='black', color='#3498db', alpha=0.7, density=True)
ax.set_title('Price Rank of Booked Hotels Within Their Search')
ax.set_xlabel('Price Rank (1 = cheapest in search)')
ax.set_ylabel('Density')
ax.axvline(x=booked_ranks.median(), color='#e74c3c', linestyle='--', label=f'Median: {booked_ranks.median():.0f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.savefig('EDA_graphs/19_booked_hotel_price_rank.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 19_booked_hotel_price_rank.png")


# DISTANCE VS BOOKING
fig, ax = plt.subplots(figsize=(10, 5))
dist_booked = dfTrain.loc[dfTrain['booking_bool'] == 1, 'orig_destination_distance'].dropna().clip(upper=5000)
dist_all = dfTrain['orig_destination_distance'].dropna().clip(upper=5000)
ax.hist(dist_all, bins=50, alpha=0.5, label='All Hotels', density=True, color='#3498db')
ax.hist(dist_booked, bins=50, alpha=0.5, label='Booked Hotels', density=True, color='#e74c3c')
ax.set_title('Origin-Destination Distance: All vs Booked')
ax.set_xlabel('Distance (clipped at 5000)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('EDA_graphs/20_distance_by_booking.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 20_distance_by_booking.png")


# AFFINITY SCORE VS BOOKING
fig, ax = plt.subplots(figsize=(10, 5))
aff_booked = dfTrain.loc[dfTrain['booking_bool'] == 1, 'srch_query_affinity_score'].dropna()
aff_all = dfTrain['srch_query_affinity_score'].dropna()
ax.hist(aff_all, bins=50, alpha=0.5, label='All Hotels', density=True, color='#3498db')
ax.hist(aff_booked, bins=50, alpha=0.5, label='Booked Hotels', density=True, color='#e74c3c')
ax.set_title('Search Query Affinity Score: All vs Booked')
ax.set_xlabel('Affinity Score (log probability)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('EDA_graphs/21_affinity_score_by_booking.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 21_affinity_score_by_booking.png")


# CLEAN UP TEMP COLUMNS
dfTrain.drop(columns=['has_history', 'price_rank'], inplace=True, errors='ignore')

print(f"\nDone! {len(os.listdir('EDA_graphs'))} graphs saved to EDA_graphs/")