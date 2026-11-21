
const DB_NAME = "TinyCoderDB";
const DB_VERSION = 1;

let database;

export function openMemory() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onupgradeneeded = event => {
            const db = event.target.result;

            db.createObjectStore("conversations", {
                keyPath: "id",
                autoIncrement: true
            });

            db.createObjectStore("training", {
                keyPath: "id",
                autoIncrement: true
            });

            db.createObjectStore("settings", {
                keyPath: "key"
            });
        };

        request.onsuccess = () => {
            database = request.result;
            resolve(database);
        };

        request.onerror = () => reject(request.error);
    });
}

export function saveRecord(storeName, record) {
    return new Promise((resolve, reject) => {
        const transaction = database.transaction(
            storeName,
            "readwrite"
        );

        const request = transaction.objectStore(storeName).add(record);

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

export function getRecords(storeName) {
    return new Promise((resolve, reject) => {
        const transaction = database.transaction(
            storeName,
            "readonly"
        );

        const request = transaction.objectStore(storeName).getAll();

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

export function saveSetting(key, value) {
    return new Promise((resolve, reject) => {
        const transaction = database.transaction(
            "settings",
            "readwrite"
        );

        const request = transaction.objectStore("settings").put({
            key,
            value
        });

        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
    });
}

export function getSetting(key) {
    return new Promise((resolve, reject) => {
        const transaction = database.transaction(
            "settings",
            "readonly"
        );

        const request = transaction.objectStore("settings").get(key);

        request.onsuccess = () =>
            resolve(request.result?.value ?? null);

        request.onerror = () => reject(request.error);
    });
}
