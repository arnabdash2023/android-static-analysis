import os
import logging
import time
from collections import Counter
import re
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Import Androguard
from androguard.misc import AnalyzeAPK

# Configure logging
logging.basicConfig(
    filename='apk_feature_extraction_full.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add console handler for real-time feedback
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Dangerous permissions
DANGEROUS_PERMISSIONS = {
    'android.permission.READ_CALENDAR',
    'android.permission.WRITE_CALENDAR',
    'android.permission.CAMERA',
    'android.permission.READ_CONTACTS',
    'android.permission.WRITE_CONTACTS',
    'android.permission.GET_ACCOUNTS',
    'android.permission.ACCESS_FINE_LOCATION',
    'android.permission.ACCESS_COARSE_LOCATION',
    'android.permission.RECORD_AUDIO',
    'android.permission.READ_PHONE_STATE',
    'android.permission.READ_PHONE_NUMBERS',
    'android.permission.CALL_PHONE',
    'android.permission.ANSWER_PHONE_CALLS',
    'android.permission.READ_CALL_LOG',
    'android.permission.WRITE_CALL_LOG',
    'android.permission.ADD_VOICEMAIL',
    'android.permission.USE_SIP',
    'android.permission.PROCESS_OUTGOING_CALLS',
    'android.permission.BODY_SENSORS',
    'android.permission.SEND_SMS',
    'android.permission.RECEIVE_SMS',
    'android.permission.READ_SMS',
    'android.permission.RECEIVE_WAP_PUSH',
    'android.permission.RECEIVE_MMS',
    'android.permission.READ_EXTERNAL_STORAGE',
    'android.permission.WRITE_EXTERNAL_STORAGE',
}

# Suspicious APIs
SUSPICIOUS_APIS = [
    'java.lang.Runtime.exec',
    'java.lang.Runtime.getRuntime',
    'java.lang.ProcessBuilder',
    'javax.crypto.Cipher',
    'javax.crypto.spec.SecretKeySpec',
    'android.telephony.SmsManager.sendTextMessage',
    'android.telephony.SmsManager.sendMultipartTextMessage',
    'java.lang.reflect.Method.invoke',
    'java.lang.Class.forName',
    'java.lang.reflect.Method',
    'java.lang.reflect.Field',
    'android.content.pm.PackageManager.setComponentEnabledSetting',
    'android.app.ActivityManager.killBackgroundProcesses',
    'android.os.PowerManager.reboot',
    'java.net.URL.openConnection',
    'java.net.HttpURLConnection',
    'org.apache.http.impl.client.DefaultHttpClient',
    'android.app.admin.DevicePolicyManager',
    'android.app.Service.startForeground',
    'android.app.DownloadManager',
    'android.app.AlarmManager',
    'android.content.Context.registerReceiver',
    'android.telephony.TelephonyManager.getDeviceId',
    'android.telephony.TelephonyManager.getSubscriberId',
    'android.telephony.TelephonyManager.getLine1Number',
    'android.telephony.TelephonyManager.getNetworkOperator',
    'android.telephony.TelephonyManager.getSimOperatorName',
    'android.location.LocationManager.getLastKnownLocation',
    'android.location.LocationManager.requestLocationUpdates',
    'android.media.AudioRecord.startRecording',
    'android.media.MediaRecorder.start',
    'android.hardware.Camera.open',
    'android.content.ContentResolver.query',
    'android.content.Context.startActivity',
    'android.content.Context.startService',
    'android.content.Context.sendBroadcast',
    'dalvik.system.DexClassLoader',
    'dalvik.system.PathClassLoader',
    'dalvik.system.InMemoryDexClassLoader',
    'java.security.MessageDigest',
    'android.provider.Settings$Secure.getString',
    'android.webkit.WebView.addJavascriptInterface',
    'android.app.NotificationManager.notify',
    'android.content.ClipboardManager',
]

# JNI method indicators
JNI_METHODS = ['JNI_OnLoad', 'Java_', 'native', 'System.loadLibrary', 'System.load']

# Crypto keywords for detection
CRYPTO_KEYWORDS = ['crypt', 'aes', 'rsa', 'sha', 'md5', 'hash', 'ssl', 'tls', 'cipher', 'encrypt', 'decrypt']

# Network-related keywords
NETWORK_KEYWORDS = ['http', 'url', 'uri', 'network', 'connect', 'socket']

def calculate_string_entropy(s):
    """Calculate Shannon entropy of a string"""
    if not s:
        return 0.0
    s = str(s)
    counts = Counter(s)
    total = len(s)
    return -sum((count / total) * np.log2(count / total) for count in counts.values())

def is_base64(s):
    """Check if a string is potentially Base64 encoded"""
    s = str(s).strip()
    if len(s) < 8 or len(s) % 4 != 0:
        return False
    return bool(re.match(r'^[A-Za-z0-9+/=]+$', s))

def safe_method_analysis(method):
    """Safely analyze a method object to access its bytecode"""
    try:
        if method and hasattr(method, 'get_code'):
            code = method.get_code()
            if code and hasattr(code, 'get_bc'):
                return code.get_bc()
    except Exception as e:
        logging.debug(f"Failed to get bytecode for method {method.get_name()}: {str(e)}")
    return None

def extract_features(apk_path, label):
    """Extract static features from an APK file with improved error handling"""
    start_time = time.time()
    features = {
        'file_name': os.path.basename(apk_path),
        'file_size': os.path.getsize(apk_path),
        'is_malware': label,
        'processing_time': 0.0,
        'error': None
    }
    
    try:
        logging.info(f"Processing {apk_path}")
        
        # Analyze the APK
        a, d, dx = AnalyzeAPK(apk_path)
        
        # Basic APK Info
        features['package_name'] = a.get_package() or ""
        features['app_name'] = a.get_app_name() or ""
        features['min_sdk'] = a.get_min_sdk_version() or 0
        features['target_sdk'] = a.get_target_sdk_version() or 0
        features['version_code'] = a.get_androidversion_code() or 0
        
        # Permissions
        permissions = a.get_permissions()
        features['permission_count'] = len(permissions)
        features['dangerous_permission_count'] = sum(1 for p in permissions if p in DANGEROUS_PERMISSIONS)
        
        # Extract specific dangerous permissions
        for perm in ['READ_SMS', 'SEND_SMS', 'CAMERA', 'READ_PHONE_STATE', 
                     'CALL_PHONE', 'ACCESS_FINE_LOCATION', 'RECORD_AUDIO', 
                     'WRITE_EXTERNAL_STORAGE']:
            full_perm = f"android.permission.{perm}"
            features[f'perm_{perm.lower()}'] = int(full_perm in permissions)
        
        # Components
        features['activity_count'] = len(a.get_activities())
        features['service_count'] = len(a.get_services())
        features['receiver_count'] = len(a.get_receivers())
        features['provider_count'] = len(a.get_providers())
        
        # Native Code
        native_libs = [f for f in a.get_files() if f.endswith('.so')]
        features['has_native_code'] = int(len(native_libs) > 0)
        features['native_lib_count'] = len(native_libs)
        native_code_size = 0
        
        for lib in native_libs:
            try:
                file_data = a.get_file(lib)
                if file_data:
                    native_code_size += len(file_data) if isinstance(file_data, bytes) else getattr(file_data, 'length', 0)
            except Exception:
                pass
                
        features['native_code_size'] = native_code_size
        
        # Code Metrics (Handle multi-DEX)
        methods = []
        if isinstance(d, list):  # Multi-DEX APK
            for dex in d:
                methods.extend(list(dex.get_methods()))
        else:  # Single DEX
            methods = list(d.get_methods())
            
        features['method_count'] = len(methods)
        features['class_count'] = len(list(dx.get_classes())) if dx else 0
        
        # Opcode Frequencies with improved error handling
        opcode_counts = Counter()
        opcodes_processed = 0
        
        for method in methods:
            bc = safe_method_analysis(method)
            if bc:
                try:
                    for inst in bc.get_instructions():
                        opcodes_processed += 1
                        opcode_counts[inst.get_name()] += 1
                except Exception as e:
                    logging.debug(f"Error processing instructions for method {method.get_name()}: {str(e)}")
                    continue
        
        # Calculate opcode diversity
        features['opcode_count'] = opcodes_processed
        features['opcode_diversity'] = len(opcode_counts)
        features['opcode_diversity_ratio'] = len(opcode_counts) / max(opcodes_processed, 1)
        
        # API and JNI Calls with improved detection
        api_calls = Counter()
        jni_calls = Counter()
        reflection_calls = Counter()
        crypto_api_calls = Counter()
        network_api_calls = Counter()
        suspicious_api_calls = Counter()
        
        # Iterate over all methods to find API calls
        for method in methods:
            try:
                # Check if the method is native (indicating JNI usage)
                if method.get_access_flags() & 0x100:  # 0x100 is the flag for 'native'
                    jni_calls[method.get_name()] += 1
                
                # Get cross-references to other methods
                for _, call_method, _ in method.get_xref_to():
                    if not hasattr(call_method, 'name') or not hasattr(call_method, 'class_name'):
                        continue
                    
                    call = f"{call_method.class_name}.{call_method.name}"
                    api_calls[call] += 1
                    
                    # Check for JNI calls (e.g., System.loadLibrary)
                    if any(jni in call.lower() for jni in JNI_METHODS):
                        jni_calls[call] += 1
                    
                    # Check for suspicious APIs
                    for sus_api in SUSPICIOUS_APIS:
                        if sus_api.lower() in call.lower():
                            suspicious_api_calls[call] += 1
                            break
                    
                    # Check for reflection
                    if "java.lang.reflect" in call or "Class.forName" in call:
                        reflection_calls[call] += 1
                    
                    # Check for crypto APIs
                    if any(keyword in call.lower() for keyword in CRYPTO_KEYWORDS):
                        crypto_api_calls[call] += 1
                    
                    # Check for network-related APIs
                    if any(net_keyword in call.lower() for net_keyword in NETWORK_KEYWORDS):
                        network_api_calls[call] += 1
            
            except Exception as e:
                logging.debug(f"Error analyzing method {method.get_name()}: {str(e)}")
                continue
        
        features['api_call_count'] = sum(api_calls.values())
        features['jni_call_count'] = sum(jni_calls.values())
        features['reflection_count'] = sum(reflection_calls.values())
        features['crypto_api_count'] = sum(crypto_api_calls.values())
        features['network_api_count'] = sum(network_api_calls.values())
        features['suspicious_api_count'] = sum(suspicious_api_calls.values())
        
        # Log top JNI calls (limited to 10 for space)
        if jni_calls:
            features['jni_calls'] = ";".join([f"{call}:{count}" for call, count in jni_calls.most_common(10)])
        else:
            features['jni_calls'] = ""
            
        # String Analysis
        strings = []
        if isinstance(d, list):
            for dex in d:
                strings.extend(dex.get_strings())
        else:
            strings = d.get_strings()
            
        features['string_count'] = len(strings)
        
        # Analyze only a sample of strings for very large APKs
        string_sample = strings[:min(5000, len(strings))]
        
        entropies = [calculate_string_entropy(s) for s in string_sample]
        features['avg_string_entropy'] = np.mean(entropies) if entropies else 0.0
        features['max_string_entropy'] = max(entropies) if entropies else 0.0
        
        # Detect potentially encrypted/encoded content
        features['base64_string_count'] = sum(1 for s in string_sample if is_base64(s))
        features['url_string_count'] = sum(1 for s in string_sample
                                         if re.search(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', str(s)))
        features['ip_address_count'] = sum(1 for s in string_sample
                                         if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', str(s)))
        
        # Obfuscation Indicators
        renamed_count = sum(1 for m in methods if re.match(r'^[a-z]{1,3}$|^[A-Z]{1,3}$', m.get_name()))
        features['renamed_method_ratio'] = renamed_count / max(len(methods), 1)
        
        # Get all classes and check for obfuscation
        all_classes = list(dx.get_classes()) if dx else []
        obfuscated_class_count = sum(1 for cls in all_classes if re.match(r'^[a-z]{1,2}(\.[a-z]{1,2})*$', cls.name))
        features['obfuscated_class_ratio'] = obfuscated_class_count / max(len(all_classes), 1)
        
        # Overall obfuscation score
        features['obfuscation_score'] = (
            features['renamed_method_ratio'] * 0.4 +
            (features['reflection_count'] / max(features['method_count'], 1)) * 0.3 +
            (features['base64_string_count'] / max(features['string_count'], 1)) * 0.3
        )
        
    except Exception as e:
        error_msg = f"Error processing {apk_path}: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        features['error'] = error_msg
    
    # Record processing time
    features['processing_time'] = time.time() - start_time
    logging.info(f"Completed {apk_path} in {features['processing_time']:.2f} seconds")
    
    return features

def process_apks(apk_dirs, labels, output_csv, max_workers=8, save_interval=50):
    """
    Process APKs in parallel and save features to CSV with intermediate saves
    """
    apk_paths = []
    apk_labels = []
    
    # Collect all APK paths and their labels
    for apk_dir, label in zip(apk_dirs, labels):
        if not os.path.exists(apk_dir):
            logging.error(f"Directory not found: {apk_dir}")
            continue
            
        for apk in os.listdir(apk_dir):
            if apk.endswith('.apk'):
                apk_path = os.path.join(apk_dir, apk)
                apk_paths.append(apk_path)
                apk_labels.append(label)
    
    total_apks = len(apk_paths)
    logging.info(f"Total APKs to process: {total_apks}")
    print(f"Total APKs to process: {total_apks}")
    
    # Check if we have previous results to resume from
    results = []
    processed_files = set()
    
    if os.path.exists(output_csv):
        try:
            previous_df = pd.read_csv(output_csv)
            results = previous_df.to_dict('records')
            processed_files = set(previous_df['file_name'].values)
            logging.info(f"Resuming from {len(processed_files)} previously processed APKs")
            print(f"Resuming from {len(processed_files)} previously processed APKs")
        except Exception as e:
            logging.error(f"Error loading previous results: {e}")
    
    # Filter out already processed files
    to_process = [(path, label) for path, label in zip(apk_paths, apk_labels) 
                  if os.path.basename(path) not in processed_files]
    
    if not to_process:
        logging.info("All APKs have been processed already.")
        print("All APKs have been processed already.")
        return pd.DataFrame(results)
    
    logging.info(f"Processing {len(to_process)} remaining APKs with {max_workers} workers")
    print(f"Processing {len(to_process)} remaining APKs with {max_workers} workers")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_apk = {
            executor.submit(extract_features, apk_path, label): (apk_path, i) 
            for i, (apk_path, label) in enumerate(to_process)
        }
        
        completed = 0
        
        for future in as_completed(future_to_apk):
            apk_path, idx = future_to_apk[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                progress_pct = completed/len(to_process)*100
                logging.info(f"Progress: {completed}/{len(to_process)} ({progress_pct:.1f}%) - Completed {os.path.basename(apk_path)}")
                print(f"Progress: {completed}/{len(to_process)} ({progress_pct:.1f}%) - Completed {os.path.basename(apk_path)}")
                
                # Save intermediate results periodically
                if completed % save_interval == 0:
                    df = pd.DataFrame(results)
                    df.to_csv(output_csv, index=False)
                    logging.info(f"Saved intermediate results: {len(results)} APKs to {output_csv}")
                    print(f"Saved intermediate results: {len(results)} APKs to {output_csv}")
                    
            except Exception as e:
                logging.error(f"Error processing future for {apk_path}: {str(e)}")
                print(f"Error processing future for {apk_path}: {str(e)}")
    
    # Final save
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        logging.info(f"Saved final results: {len(df)} APKs to {output_csv}")
        print(f"Saved final results: {len(df)} APKs to {output_csv}")
        
        # Additional statistics
        successful = df[df['error'].isna()].shape[0]
        failed = df[df['error'].notna()].shape[0]
        logging.info(f"Successfully processed: {successful} APKs")
        logging.info(f"Failed to process: {failed} APKs")
        print(f"Successfully processed: {successful} APKs")
        print(f"Failed to process: {failed} APKs")
    else:
        logging.error("No results to save.")
        print("No results to save.")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from APK files for malware detection')
    parser.add_argument('--benign', required=True, help='Directory containing benign APK files')
    parser.add_argument('--malware', required=True, help='Directory containing malware APK files')
    parser.add_argument('--output', default='apk_features.csv', help='Output CSV file')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes')
    parser.add_argument('--save-interval', type=int, default=50, help='Save intermediate results every N APKs')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.benign):
        print(f"Error: Benign directory not found: {args.benign}")
        exit(1)
    
    if not os.path.exists(args.malware):
        print(f"Error: Malware directory not found: {args.malware}")
        exit(1)
    
    print(f"Starting feature extraction from {args.benign} and {args.malware}")
    print(f"Results will be saved to {args.output}")
    
    # Process APKs
    start_time = time.time()
    df = process_apks(
        [args.benign, args.malware], 
        [0, 1], 
        args.output,
        max_workers=args.workers,
        save_interval=args.save_interval
    )
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per APK: {total_time/max(len(df), 1):.2f} seconds")
    print(f"Results saved to {args.output}")