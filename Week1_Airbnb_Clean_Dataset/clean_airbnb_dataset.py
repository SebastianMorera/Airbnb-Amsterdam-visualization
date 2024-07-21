import numpy as np
from numpy import genfromtxt
from currency_converter import CurrencyConverter
import math
import timeit


# Preprocessing the Dataset #
def preprocessing_dataset():
    # For readability purposes, we will disable scientific notation for numbers
    np.set_printoptions(suppress=True)

    # Step 1
    initial_dataset = step1_find_the_delimiter()

    # Step 2
    matrix_step2 = step2_clean_it_up(initial_dataset)

    # Step 3
    matrix_step3 = step3_wide_to_long(matrix_step2)

    # Step 4 and 5
    matrix_step4 = step4and5_remove_inappropriate_characters_and_verify(matrix_step3)

    # Step 6
    matrix_step6 = step6_enabling_numerical_operations(matrix_step4)

    return matrix_step6


def step1_find_the_delimiter():
    # Find the delimiter used in the csv file
    # You can either find it by manually looking at the csv file or by using terminal command to see the file:
    # Command: cat Initial_Week1_Airbnb_Amsterdam_Dataset.csv
    filename = 'Initial_Week1_Airbnb_Amsterdam_Dataset.csv'
    initial_dataset = genfromtxt(filename, delimiter="|", dtype="unicode")

    # Output the first four columns for inspection to see if you've got the data formatted how you'd like.
    print("Step 1: Data formated with the delimiter")
    print(initial_dataset[:, :4], "\n")

    return initial_dataset


def step2_clean_it_up(dataset):
    # In order for your calculations to run correctly, you need to have only the "relevant" numbers/entries present in
    # your dataset. This means no headers, footers, redundant IDs, etc. You need to remove the first row and column
    matrix = dataset[1:, 1:]

    # Verify your work by again by printing out the first four columns.
    print("Step 2: Clean the first row and column to remove headers, keep only the 'relevant' numbers/entries")
    print(matrix[:, :4], "\n")

    return matrix


def step3_wide_to_long(matrix):
    # The dataset is shifted by 90 degrees. Let's shift it another 90 degrees to get it back to how we'd expect,
    # which is in a much more readable format.
    matrix = matrix.T

    # Verify your work by printing out of the first five rows.
    print("Step 3: The dataset is shifted by 90 degrees. Shift it by another 90 degrees to get a more readable format.")
    print(matrix[:5, :], "\n")

    return matrix


def step4and5_remove_inappropriate_characters_and_verify(matrix):
    # String characters like commas and dollar signs are yet again present in the dataset. We need ro remove them
    # The dollar sign is for the prices and the comma is where prices are big above 1000.
    # Remove the dollar sign
    matrix = np.char.replace(matrix, "$", "")
    # Remove the comma
    matrix = np.char.replace(matrix, ",", "")

    print("Step 4 and 5: Remove inappropriate characters and verify they are all removed")
    # Check if the dollar sign is in our dataset
    print("For dollar sign: \n", matrix[np.char.find(matrix, "$") > -1], "\n")
    # Check if commas are in our dataset
    print("For commas: \n", matrix[np.char.find(matrix, ",") > -1], "\n")

    return matrix


def step6_enabling_numerical_operations(matrix):
    # Enabling numerical operations (calculations) requires you to change the dtype from string/Unicode characters to
    # float of 32-bit precision.
    # Change Unicode to float32
    matrix = matrix.astype(np.float32)

    # Print out the first five rows (and inspect the dtype for correctness)
    print("Step 6: Enabling numerical operations (calculations) requires to change to float of 32-bit precision.")
    print(matrix[:5, :], "\n")

    return matrix


# The price is right #
def the_price_is_right(matrix):
    # Since all our values in the matrix are now recognized as numbers, we can perform some awesome calculations!
    # Our next objective is to change the currency from US dollars to another currency. This can be any currency you
    # like, except for the US dollar. The currency conversion calculations you'll be performing should be applied
    # to the second column only (the price).
    cc = CurrencyConverter()

    # Currency converter library has a total of 42 currencies. Please select one of them, and use it to convert the
    # dollars into your chosen currency. You can check which are available by running:
    print("Currency converter library has a total of 42 currencies, here is the list: ")
    print(cc.currencies, "\n")

    # Step 7
    matrix_step7 = step7_pick_any_currency_to_convert(cc, matrix)

    # Step 8
    matrix_step8 = step8_inflation(matrix_step7)

    # Step 9
    matrix_step9 = step9_too_many_decimals(matrix_step8)

    return matrix_step9


def step7_pick_any_currency_to_convert(cc, matrix):
    print("Step 7: Pick any currency and convert the US dollar to the currency of your choice")
    print("Before the currency conversion: \n", matrix[:, 1], "\n")
    # Get the rate of conversion from the US dollar to your currency of choice
    price_rate = cc.convert(1, "USD", "CAD")

    # Multiply the dollar column by your currency of choice
    matrix[:, 1] = matrix[:, 1] * price_rate

    # Output the second column to verify the prices were converted
    print("After the currency conversion: \n", matrix[:, 1], "\n")
    return matrix


def step8_inflation(matrix):
    # Recent inflation all around the world has caused many companies to raise their prices. Consequently, Airbnb
    # listings have also raised their prices by a certain amount. Use 7% of inflation rate for the prices.
    matrix[:, 1] = matrix[:, 1] * 1.07

    # Output the second column to verify the prices were updated with the inflation rate
    print("Step 8: Airbnb listings have raised their prices. Use 7% of inflation rate for the prices")
    print(matrix[:, 1], "\n")

    return matrix


def step9_too_many_decimals(matrix):
    # You might have some prices longer than two decimals after changing the currency and adjusting the price for
    # inflation. Please round the prices down to the nearest two decimals using a NumPy native function.
    matrix[:, 1] = np.round(matrix[:, 1], 2)

    # Output the second column to verify the prices were rounded to the nearest two decimals
    print("Step 9: Too many decimals for the prices. Round the prices down to the nearest two decimals")
    print(matrix[:, 1], "\n")

    return matrix


# Where u (want to be) at #
def where_u_at():
    # Favorite location. List your coordinates as floats. Example latitude = 52.3600, longitude = 4.8852
    # The location chosen: Brouwerij ‘t IJ (historic brewery)
    latitude = 52.3667
    longitude = 4.9264

    print("Step 10: Choose a location : \n", "Latitude = ", latitude, " Longitude = ", longitude, "\n")

    return latitude, longitude


# Listing All Listings #
def list_all_listing_with_vectorization(latitude, longitude, matrix):
    # Allow a Python function to be used in a (semi-)vectorized way
    conv_to_meters = np.vectorize(from_location_to_airbnb_listing_in_meters)

    # Apply the function
    start_time = timeit.default_timer()  # Start time measurement
    conv_to_meters(latitude, longitude, matrix[:, 2], matrix[:, 3])
    elapsed_time = (timeit.default_timer() - start_time) * 1000  # Convert to milliseconds

    print("Step 11: Calculate the best listings for you based on the locations you were planning to visit.")
    print(f"Time taken (milliseconds): {elapsed_time:.4f}", "ms", "\n")  # Print time with 4 decimal places


def from_location_to_airbnb_listing_in_meters(lat1: float, lon1: float, lat2: list, lon2: list):
    # Source: https://community.esri.com/t5/coordinate-reference-systems-blog
    # /distance-on-a-sphere-the-haversine-formula/ba-p/902128

    r = 6371000  # Radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = r * c  # Output distance in meters

    return round(meters, 0)


# Do it faster #
def how_much_faster(latitude, longitude, matrix):
    # Instead of using the from_location_to_airbnb_listing_in_meters function, convert the function into a pure Numpy
    # function.
    print("Step 12: Convert the from_location_to_airbnb_listing_in_meters function into a pure Numpy function \n")

    # Apply the function
    start_time = timeit.default_timer()  # Start time measurement
    from_location_to_airbnb_listing_in_meters_numpy_version(latitude, longitude, matrix[:, 2], matrix[:, 3])
    elapsed_time = (timeit.default_timer() - start_time) * 1000000  # Convert to microseconds

    print("Step 13: Calculate the best listings for you based on the locations using pure Numpy functions")
    print(f"Time taken (microseconds): {elapsed_time:.4f}", "µs", "\n")  # Print time with 4 decimal places


def from_location_to_airbnb_listing_in_meters_numpy_version(lat1: float, lon1: float, lat2: np.ndarray,
                                                            lon2: np.ndarray):
    r = 6371000  # Radius of Earth in meters
    phi_1 = np.radians(lat1)  # CHANGE THIS
    phi_2 = np.radians(lat2)  # CHANGE THIS

    delta_phi = np.radians(lat2 - lat1)  # CHANGE THIS
    delta_lambda = np.radians(lon2 - lon1)  # CHANGE THIS

    a = (
            np.sin(delta_phi / 2.0) ** 2  # CHANGE THIS
            + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2.0) ** 2  # CHANGE THIS (3x)
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))  # CHANGE THIS (3x)

    meters = r * c  # Output distance in meters

    return np.round(meters, 0)  # CHANGE THIS


# Prep the dataset for download
def prep_dataset_for_download(latitude, longitude, matrix):
    # Run the previous method
    meters = from_location_to_airbnb_listing_in_meters_numpy_version(latitude, longitude, matrix[:, 2], matrix[:, 3])

    # Add an axis to make concatenation possible
    meters = meters.reshape(-1, 1)

    # Append the distance in meters to the matrix
    matrix = np.concatenate((matrix, meters), axis=1)

    # Append a color to the matrix
    colors = np.zeros(meters.shape)
    matrix = np.concatenate((matrix, colors), axis=1)

    # Append our entry to the matrix
    fav_entry = np.array([1, 0, 52.3667, 4.9264, 0, 1]).reshape(1, -1)  # Changed for Brewery location
    matrix = np.concatenate((fav_entry, matrix), axis=0)

    # Export the data to use in the primer for next week
    np.savetxt("Final_Week1_Airbnb_Amsterdam_Dataset.csv", matrix, delimiter=",")


if __name__ == '__main__':
    preprocessed_matrix = preprocessing_dataset()
    price_is_right_matrix = the_price_is_right(preprocessed_matrix)
    desired_latitude, desired_longitude = where_u_at()
    list_all_listing_with_vectorization(desired_latitude, desired_longitude, price_is_right_matrix)
    how_much_faster(desired_latitude, desired_longitude, price_is_right_matrix)
    prep_dataset_for_download(desired_latitude, desired_longitude, price_is_right_matrix)
