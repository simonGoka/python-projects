import pyfiglet
from termcolor import colored

def flyer():
    banner = pyfiglet.figlet_format("HOSPITAL MANAGEMENT SYSTEM")
    print(colored(banner, "red"))

the_data_base = {}

def add_patient():
    patient_id = input("Enter patient ID: ")
    name = input("Enter patient name: ")
    age = input("Enter patient age: ")
    the_data_base[patient_id] = {'name': name, 'age': age}
    print(f"Added Patient {name} with ID: {patient_id} and age: {age}")

def delete_patient():
    patient_id = input("Enter patient ID to delete: ")
    if patient_id in the_data_base:
        del the_data_base[patient_id]
        print(f"Patient {patient_id} deleted.")
    else:
        print(f"Patient {patient_id} not found.")

def update_patient():
    patient_id = input("Enter patient ID to update: ")
    if patient_id in the_data_base:
        name = input("Enter new patient name: ")
        age = input("Enter new patient age: ")
        the_data_base[patient_id] = {'name': name, 'age': age}
        print(f"Patient {patient_id} updated.")
    else:
        print(f"Patient {patient_id} not found.")

def view_patients():
    if not the_data_base:
        print("No patients found.")
    else:
        for patient_id, details in the_data_base.items():
            print(f"ID: {patient_id}, Name: {details['name']}, Age: {details['age']}")

def search_patient():
    patient_id = input("Enter patient ID to search: ")
    if patient_id in the_data_base:
        details = the_data_base[patient_id]
        print(f"ID: {patient_id}, Name: {details['name']}, Age: {details['age']}")
    else:
        print(f"Patient {patient_id} not found.")

def main():
    flyer()
    while True:
        print("\nHospital Patient Management System")
        print("1. Add Patient")
        print("2. Delete Patient")
        print("3. Update Patient")
        print("4. View Patients")
        print("5. Search Patient")
        print("6. Exit")

        choice = input("Enter your choice: ")
        if choice == '1':
            add_patient()
        elif choice == '2':
            delete_patient()
        elif choice == '3':
            update_patient()
        elif choice == '4':
            view_patients()
        elif choice == '5':
            search_patient()
        elif choice == '6':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()