#!/usr/bin/env python3
"""
🚀 LANCEUR EXPÉRIENCES CORRIGÉ
SDPA vs Flash Attention - Version sans conflits conda/pip
Développé par Kennedy Kitoko 🇨🇩
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

class FixedExperimentLauncher:
    """
    Lanceur corrigé sans conflits conda/pip
    """
    
    def __init__(self):
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = f"experiments_session_{self.session_id}"
        self.results = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': datetime.now().isoformat(),
                'researcher': 'Kennedy Kitoko 🇨🇩',
                'experiment_type': 'SDPA vs Flash Attention Scientific Comparison'
            },
            'validation_results': {},
            'sdpa_experiment': {},
            'flash_experiment': {},
            'session_summary': {}
        }
        
        # Création structure session
        os.makedirs(self.base_dir, exist_ok=True)
        
        print(f"🚀 Lanceur d'expériences corrigé initialisé")
        print(f"📁 Session: {self.session_id}")
        print(f"📂 Dossier: {self.base_dir}")
    
    def conda_aware_validation(self):
        """Validation compatible conda sans internet"""
        print("\n🔍 VALIDATION CONDA-AWARE...")
        
        validation = {
            'python_ok': False,
            'torch_ok': False,
            'cuda_ok': False,
            'ultralytics_ok': False,
            'flash_attn_ok': False,
            'pillow_ok': False,
            'dataset_ok': False
        }
        
        # Python
        if sys.version_info >= (3, 8):
            validation['python_ok'] = True
            print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
        
        # PyTorch + CUDA
        try:
            import torch
            validation['torch_ok'] = True
            print(f"✅ PyTorch {torch.__version__}")
            
            if torch.cuda.is_available():
                validation['cuda_ok'] = True
                print(f"✅ CUDA - {torch.cuda.get_device_name(0)}")
        except ImportError:
            print("❌ PyTorch non disponible")
        
        # Ultralytics
        try:
            from ultralytics import YOLO
            validation['ultralytics_ok'] = True
            print("✅ Ultralytics YOLO")
        except ImportError:
            print("❌ Ultralytics non disponible")
        
        # Flash Attention
        try:
            import flash_attn
            validation['flash_attn_ok'] = True
            print(f"✅ Flash Attention {getattr(flash_attn, '__version__', 'OK')}")
        except ImportError:
            print("⚠️ Flash Attention non disponible (non bloquant)")
        
        # Pillow (test conda-aware)
        try:
            from PIL import Image
            validation['pillow_ok'] = True
            print("✅ Pillow (PIL) disponible")
        except ImportError:
            print("⚠️ Pillow non accessible")
        
        # Dataset
        dataset_paths = ['Weeds-3', './Weeds-3', '../Weeds-3']
        for path in dataset_paths:
            if os.path.exists(os.path.join(path, 'train', 'images')):
                validation['dataset_ok'] = True
                print(f"✅ Dataset: {os.path.abspath(path)}")
                break
        
        if not validation['dataset_ok']:
            print("⚠️ Dataset non trouvé")
        
        # Résumé
        critical_checks = ['python_ok', 'torch_ok', 'cuda_ok', 'ultralytics_ok']
        critical_passed = sum(validation[check] for check in critical_checks)
        
        ready = critical_passed >= 4
        
        print(f"\n📊 Validation: {sum(validation.values())}/{len(validation)} checks")
        print(f"🎯 Critique: {critical_passed}/{len(critical_checks)} essentiels")
        
        self.results['validation_results'] = {
            'checks': validation,
            'ready_for_experiments': ready,
            'critical_passed': critical_passed
        }
        
        return ready
    
    def run_sdpa_experiment_fixed(self):
        """Expérience SDPA sans erreurs ComprehensiveMonitor"""
        print("\n🔬 EXPÉRIENCE SDPA CORRIGÉE...")
        
        try:
            # Configuration SDPA
            config = {
                'model': 'yolo12n.pt',
                'data': 'Weeds-3/data.yaml',
                'epochs': 100,  # Réduit pour test
                'batch': 8,
                'imgsz': 640,
                'device': 'cuda:0',
                'amp': False,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'patience': 10,
                'project': self.base_dir,
                'name': 'sdpa_experiment'
            }
            
            print("📋 Configuration SDPA:")
            for k, v in config.items():
                print(f"   {k}: {v}")
            
            # Import et monitoring simple
            from ultralytics import YOLO
            
            # Monitoring basique (sans ComprehensiveMonitor complexe)
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            print("🚀 Démarrage SDPA avec PyTorch natif...")
            
            # Variables environnement SDPA
            os.environ['TORCH_CUDNN_BENCHMARK'] = '1'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Chargement modèle
            print(config['model'])
            model = YOLO(config['model'])
            
            # Entraînement
            results = model.train(**{k: v for k, v in config.items() if k not in ['model']})
            
            # Métriques finales
            end_time = time.time()
            end_memory = self.get_memory_usage()
            duration = end_time - start_time
            
            self.results['sdpa_experiment'] = {
                'success': True,
                'duration_seconds': duration,
                'duration_minutes': duration / 60,
                'config': config,
                'results_path': str(results.save_dir) if hasattr(results, 'save_dir') else None,
                'memory_start_mb': start_memory,
                'memory_end_mb': end_memory,
                'memory_used_mb': end_memory - start_memory
            }
            
            print(f"✅ SDPA terminé en {duration/60:.1f} minutes")
            print(f"📊 Mémoire utilisée: {end_memory - start_memory:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur SDPA: {e}")
            self.results['sdpa_experiment'] = {
                'success': False,
                'error': str(e),
                'duration_seconds': time.time() - start_time if 'start_time' in locals() else 0
            }
            return False
    
    def run_flash_experiment_fixed(self):
        """Expérience Flash Attention corrigée"""
        print("\n🔥 EXPÉRIENCE FLASH ATTENTION CORRIGÉE...")
        
        try:
            # Test Flash Attention
            try:
                import flash_attn
                print(f"✅ Flash Attention v{getattr(flash_attn, '__version__', 'unknown')}")
            except ImportError:
                print("❌ Flash Attention non disponible")
                self.results['flash_experiment'] = {
                    'success': False,
                    'error': 'Flash Attention not available',
                    'note': 'SDPA seul disponible'
                }
                return False
            
            # Configuration identique SDPA
            config = {
                'model': 'yolo12n.pt',
                'data': 'Weeds-3/data.yaml',
                'epochs': 100,  # Identique SDPA
                'batch': 8,
                'imgsz': 640,
                'device': 'cuda:0',
                'amp': False,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'patience': 10,
                'project': self.base_dir,
                'name': 'flash_experiment'
            }
            
            print("📋 Configuration Flash Attention:")
            for k, v in config.items():
                print(f"   {k}: {v}")
            
            # Variables Flash Attention
            os.environ['FLASH_ATTENTION_FORCE_USE'] = '1'
            os.environ['FLASH_ATTENTION_SKIP_CUDA_BUILD'] = '0'
            
            from ultralytics import YOLO
            
            # Monitoring basique
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            print("🚀 Démarrage Flash Attention...")
            
            # Chargement modèle
            model = YOLO(config['model'])
            
            # Entraînement
            results = model.train(**{k: v for k, v in config.items() if k not in ['model']})
            
            # Métriques finales
            end_time = time.time()
            end_memory = self.get_memory_usage()
            duration = end_time - start_time
            
            self.results['flash_experiment'] = {
                'success': True,
                'duration_seconds': duration,
                'duration_minutes': duration / 60,
                'config': config,
                'results_path': str(results.save_dir) if hasattr(results, 'save_dir') else None,
                'memory_start_mb': start_memory,
                'memory_end_mb': end_memory,
                'memory_used_mb': end_memory - start_memory
            }
            
            print(f"✅ Flash Attention terminé en {duration/60:.1f} minutes")
            print(f"📊 Mémoire utilisée: {end_memory - start_memory:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur Flash Attention: {e}")
            self.results['flash_experiment'] = {
                'success': False,
                'error': str(e),
                'duration_seconds': time.time() - start_time if 'start_time' in locals() else 0
            }
            return False
    
    def get_memory_usage(self):
        """Utilisation mémoire actuelle"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0
    
    def simple_comparison_analysis(self):
        """Analyse comparative simple"""
        print("\n📊 ANALYSE COMPARATIVE SIMPLE...")
        
        try:
            sdpa_success = self.results.get('sdpa_experiment', {}).get('success', False)
            flash_success = self.results.get('flash_experiment', {}).get('success', False)
            
            comparison = {
                'timestamp': datetime.now().isoformat(),
                'experiments_compared': 0,
                'sdpa_available': sdpa_success,
                'flash_available': flash_success,
                'comparison_possible': sdpa_success or flash_success
            }
            
            if sdpa_success:
                comparison['experiments_compared'] += 1
                sdpa_data = self.results['sdpa_experiment']
                comparison['sdpa_metrics'] = {
                    'duration_minutes': sdpa_data['duration_minutes'],
                    'memory_used_mb': sdpa_data.get('memory_used_mb', 0),
                    'results_path': sdpa_data.get('results_path')
                }
            
            if flash_success:
                comparison['experiments_compared'] += 1
                flash_data = self.results['flash_experiment']
                comparison['flash_metrics'] = {
                    'duration_minutes': flash_data['duration_minutes'],
                    'memory_used_mb': flash_data.get('memory_used_mb', 0),
                    'results_path': flash_data.get('results_path')
                }
            
            # Comparaison si les deux disponibles
            if sdpa_success and flash_success:
                sdpa_time = self.results['sdpa_experiment']['duration_minutes']
                flash_time = self.results['flash_experiment']['duration_minutes']
                
                comparison['performance_comparison'] = {
                    'sdpa_faster': sdpa_time < flash_time,
                    'time_difference_minutes': abs(sdpa_time - flash_time),
                    'speed_advantage': 'SDPA' if sdpa_time < flash_time else 'Flash Attention'
                }
            
            # Sauvegarde analyse
            analysis_file = f"{self.base_dir}/simple_comparison_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(comparison, f, indent=4, default=str)
            
            print(f"✅ Analyse comparative terminée")
            print(f"📊 Expériences comparées: {comparison['experiments_compared']}")
            print(f"📁 Analyse sauvegardée: {analysis_file}")
            
            return comparison
            
        except Exception as e:
            print(f"❌ Erreur analyse: {e}")
            return {}
    
    def generate_final_report(self):
        """Rapport final sans erreurs"""
        print("\n📋 GÉNÉRATION RAPPORT FINAL...")
        
        try:
            session_end = datetime.now()
            session_start = datetime.fromisoformat(self.results['session_info']['start_time'])
            total_duration = (session_end - session_start).total_seconds()
            
            # Résumé session
            summary = {
                'session_duration_minutes': total_duration / 60,
                'sdpa_success': self.results.get('sdpa_experiment', {}).get('success', False),
                'flash_success': self.results.get('flash_experiment', {}).get('success', False),
                'validation_success': self.results.get('validation_results', {}).get('ready_for_experiments', False)
            }
            
            summary['experiments_completed'] = sum([summary['sdpa_success'], summary['flash_success']])
            summary['research_success'] = summary['experiments_completed'] > 0
            
            # Sauvegarde résultats complets
            self.results['session_summary'] = summary
            results_file = f"{self.base_dir}/complete_session_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=4, default=str)
            
            # Rapport lisible
            report_file = f"{self.base_dir}/session_summary_report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("🚀 RAPPORT SESSION EXPÉRIMENTALE CORRIGÉE\n")
                f.write("=" * 50 + "\n")
                f.write(f"Session: {self.session_id}\n")
                f.write(f"Durée: {summary['session_duration_minutes']:.1f} minutes\n")
                f.write(f"Date: {session_end.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("📊 RÉSULTATS EXPÉRIENCES:\n")
                f.write(f"SDPA: {'✅ Réussi' if summary['sdpa_success'] else '❌ Échoué'}\n")
                f.write(f"Flash Attention: {'✅ Réussi' if summary['flash_success'] else '❌ Échoué'}\n")
                f.write(f"Total réussies: {summary['experiments_completed']}/2\n\n")
                
                if summary['sdpa_success']:
                    sdpa_data = self.results['sdpa_experiment']
                    f.write(f"SDPA - Durée: {sdpa_data['duration_minutes']:.1f} min\n")
                    f.write(f"SDPA - Résultats: {sdpa_data.get('results_path', 'N/A')}\n")
                
                if summary['flash_success']:
                    flash_data = self.results['flash_experiment']
                    f.write(f"Flash - Durée: {flash_data['duration_minutes']:.1f} min\n")
                    f.write(f"Flash - Résultats: {flash_data.get('results_path', 'N/A')}\n")
                
                f.write(f"\n📁 Dossier session: {self.base_dir}\n")
                f.write("🇨🇩 Kennedy Kitoko - Agricultural AI Innovation\n")
            
            print(f"💾 Rapport final: {report_file}")
            
            return summary
            
        except Exception as e:
            print(f"❌ Erreur rapport final: {e}")
            return {}
    
    def run_fixed_workflow(self):
        """Workflow corrigé sans erreurs conda/pip"""
        print("🚀 WORKFLOW CORRIGÉ - SANS CONFLITS CONDA/PIP")
        print("🇨🇩 Kennedy Kitoko - Agricultural AI Innovation")
        print("=" * 60)
        
        try:
            # Phase 1: Validation conda-aware
            print(f"\n{'='*20}")
            print("PHASE 1: VALIDATION CONDA-AWARE")
            print(f"{'='*20}")
            
            if not self.conda_aware_validation():
                print("❌ Validation échouée - Arrêt")
                return False
            
            # Phase 2: SDPA
            print(f"\n{'='*20}")
            print("PHASE 2: EXPÉRIENCE SDPA")
            print(f"{'='*20}")
            
            sdpa_success = self.run_sdpa_experiment_fixed()
            
            # Phase 3: Flash Attention
            print(f"\n{'='*20}")
            print("PHASE 3: EXPÉRIENCE FLASH ATTENTION")
            print(f"{'='*20}")
            
            flash_success = self.run_flash_experiment_fixed()
            
            # Phase 4: Analyse
            print(f"\n{'='*20}")
            print("PHASE 4: ANALYSE COMPARATIVE")
            print(f"{'='*20}")
            
            self.simple_comparison_analysis()
            
            # Phase 5: Rapport final
            print(f"\n{'='*20}")
            print("PHASE 5: RAPPORT FINAL")
            print(f"{'='*20}")
            
            summary = self.generate_final_report()
            
            # Résumé final
            print(f"\n{'='*30}")
            print("🎉 WORKFLOW CORRIGÉ TERMINÉ")
            print(f"{'='*30}")
            
            if summary.get('research_success', False):
                print("🏆 SUCCÈS! Au moins une expérience réussie")
                print("📊 Données scientifiques collectées")
            else:
                print("⚠️ Aucune expérience complétée")
            
            print(f"📁 Tous les résultats dans: {self.base_dir}")
            
            return summary.get('research_success', False)
            
        except Exception as e:
            print(f"❌ Erreur workflow: {e}")
            return False

def main():
    """Point d'entrée principal corrigé"""
    print("🚀 LANCEUR EXPÉRIENCES CORRIGÉ")
    print("🎯 SDPA vs Flash Attention - Sans conflits conda/pip")
    print("🇨🇩 Kennedy Kitoko - AI Democratization")
    print("=" * 60)
    
    print("\n🔧 Corrections appliquées:")
    print("✅ Validation compatible conda")
    print("✅ Pas de conflits pip/conda pour Pillow")
    print("✅ Monitoring simplifié sans erreurs")
    print("✅ Pas de tests internet inutiles")
    
    print(f"\n⏱️ Durée estimée: 1-2 heures (epochs réduits)")
    print(f"💾 Espace requis: ~3 GB")
    
    # Confirmation
    try:
        confirm = input("\n🚀 Lancer le workflow corrigé? (Y/n): ").strip().lower()
        if confirm in ['n', 'no', 'non']:
            print("❌ Workflow annulé")
            return False
    except KeyboardInterrupt:
        print("\n❌ Workflow annulé")
        return False
    
    # Lancement
    launcher = FixedExperimentLauncher()
    
    try:
        success = launcher.run_fixed_workflow()
        
        if success:
            print("\n🏆 WORKFLOW RÉUSSI!")
            print("📊 Comparaison SDPA vs Flash Attention terminée")
        else:
            print("\n📋 WORKFLOW PARTIEL")
            print("💾 Certaines données collectées")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Erreur workflow: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    print(f"\n{'='*50}")
    print("🇨🇩 Kennedy Kitoko - Agricultural AI Innovation")
    print("🌍 Democratizing AI for Global Agriculture")
    print("📚 SDPA: Simplicity Achieving Excellence")
    print(f"{'='*50}")
    
    sys.exit(0 if success else 1)