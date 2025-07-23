// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Strangeness-to-collision association tests
//
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"
#include "PWGLF/DataModel/LFStrangenessMLTables.h"
#include "PWGLF/DataModel/LFParticleIdentification.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/McCollisionExtra.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/PIDResponse.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "CCDB/BasicCCDBManager.h"
#include "PWGLF/Utils/strangenessBuilderHelper.h"
#include "Common/Core/TPCVDriftManager.h"
#include "Tools/ML/MlResponse.h"
#include "Tools/ML/model.h"

#include <TFile.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <Math/Vector4D.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>
#include <cmath>
#include <array>
#include <cstdlib>
#include "Framework/ASoAHelpers.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::ml;
using std::array;
using std::vector;
using std::pair;
using std::make_pair;

// using MyTracks = soa::Join<aod::Tracks, aod::TracksExtra, aod::pidTPCPr>;
using TracksCompleteIU = soa::Join<aod::TracksIU, aod::TracksExtra, aod::TracksCovIU, aod::TracksDCA>;
using TracksCompleteIUMC = soa::Join<aod::TracksIU, aod::TracksExtra, aod::TracksCovIU, aod::TracksDCA, aod::McTrackLabels>;
using V0DataLabeled = soa::Join<aod::V0Datas, aod::McV0Labels>;
using CascMC = soa::Join<aod::CascDataExt, aod::McCascLabels>;
using TraCascMC = soa::Join<aod::TraCascDatas, aod::McTraCascLabels>;
using RecoedMCCollisions = soa::Join<aod::McCollisions, aod::McCollsExtra>;
using CollisionsWithEvSels = soa::Join<aod::Collisions, aod::EvSels>;

// For MC association in pre-selection
using LabeledTracksExtra = soa::Join<aod::TracksIU, aod::TracksCovIU, aod::TracksExtra, aod::McTrackLabels>;

struct v0assoqa {  
  o2::ml::OnnxModel deduplication_bdt; 

  Produces<aod::V0Duplicates> photonDuplicates;  
  
  std::map<std::string, std::string> metadata;
  Configurable<std::string> ccdbUrl{"ccdb-url", "http://alice-ccdb.cern.ch", "url of the ccdb repository"};
  Configurable<std::string> BDTLocalPath{"BDTLocalPath", "Deduplication_BDTModel.onnx", "(std::string) Path to the local .onnx file."};
  Configurable<std::string> BDTPathCCDB{"BDTPathCCDB", "Users/g/gsetouel/MLModels2", "Path on CCDB"};
  Configurable<int64_t> timestampCCDB{"timestampCCDB", -1, "timestamp of the ONNX file for ML model used to query in CCDB.  Exceptions: > 0 for the specific timestamp, 0 gets the run dependent timestamp"};
  Configurable<bool> loadModelsFromCCDB{"loadModelsFromCCDB", false, "Flag to enable or disable the loading of models from CCDB"};
  Configurable<bool> enableOptimizations{"enableOptimizations", false, "Enables the ONNX extended model-optimization: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED)"};
  Configurable<bool> PredictV0Association{"PredictV0Association", false, "Flag to enable or disable the loading of model"};
  Configurable<float> BDTthreshold{"BDTthreshold", -1, "BDT threshold for deduplication. If -1, no threshold is applied."};

  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  // helper object
  o2::pwglf::strangenessBuilderHelper straHelper;

  o2::ccdb::CcdbApi ccdbApi;
  Service<o2::ccdb::BasicCCDBManager> ccdb;

  int mRunNumber;
  o2::base::MatLayerCylSet* lut = nullptr;

  // for handling TPC-only tracks (photons)
  o2::aod::common::TPCVDriftManager mVDriftMgr;

  // ML for deduplication 
  Configurable<bool> fillDuplicatesTable{"fillDuplicatesTable", false, "if true, fill table with duplicated v0s"};

  // CCDB options
  struct : ConfigurableGroup {
    std::string prefix = "ccdb";
    Configurable<std::string> ccdburl{"ccdb-url", "http://alice-ccdb.cern.ch", "url of the ccdb repository"};
    Configurable<std::string> grpPath{"grpPath", "GLO/GRP/GRP", "Path of the grp file"};
    Configurable<std::string> grpmagPath{"grpmagPath", "GLO/Config/GRPMagField", "CCDB path of the GRPMagField object"};
    Configurable<std::string> lutPath{"lutPath", "GLO/Param/MatLUT", "Path of the Lut parametrization"};
    Configurable<std::string> geoPath{"geoPath", "GLO/Config/GeometryAligned", "Path of the geometry file"};
  } ccdbConfigurations;

  // V0 building options
  struct : ConfigurableGroup {
    std::string prefix = "v0BuilderOpts";
    Configurable<bool> moveTPCOnlyTracks{"moveTPCOnlyTracks", true, "if dealing with TPC-only tracks, move them according to TPC drift / time info"};
    Configurable<bool> skipNonTPCOnly{"skipNonTPCOnly", false, "only analyze full TPC-only v0s"};
    Configurable<bool> selectBuiltOnly{"selectBuiltOnly", true, "if true, select only built v0s"};
    Configurable<bool> selectPrimariesOnly{"selectPrimariesOnly", true, "if true, select only primaries"};

    // baseline conditionals of V0 building
    Configurable<int> minCrossedRows{"minCrossedRows", -1, "minimum TPC crossed rows for daughter tracks"};
    Configurable<float> dcanegtopv{"dcanegtopv", .0, "DCA Neg To PV"};
    Configurable<float> dcapostopv{"dcapostopv", .0, "DCA Pos To PV"};
    Configurable<double> v0cospa{"v0cospa", -2, "V0 CosPA"}; // double -> N.B. dcos(x)/dx = 0 at x=0)
    Configurable<float> dcav0dau{"dcav0dau", 10000.0, "DCA V0 Daughters"};
    Configurable<float> v0radius{"v0radius", 0.0, "v0radius"};
    Configurable<float> maxDaughterEta{"maxDaughterEta", 5.0, "Maximum daughter eta (in abs value)"};
    Configurable<float> PhotonMaxMass{"PhotonMaxMass", 0.1, "Max photon mass (GeV/c^{2})"};
    Configurable<float> V0Rapidity{"V0Rapidity", 0.5, "Max v0 rapidity"};
  } v0BuilderOpts;

  // cascade building options
  struct : ConfigurableGroup {
    std::string prefix = "cascadeBuilderOpts";
    // conditionals
    Configurable<int> minCrossedRows{"minCrossedRows", 50, "minimum TPC crossed rows for daughter tracks"};
    Configurable<float> dcabachtopv{"dcabachtopv", .05, "DCA Bach To PV"};
    Configurable<float> cascradius{"cascradius", 0.9, "cascradius"};
    Configurable<float> casccospa{"casccospa", 0.95, "casccospa"};
    Configurable<float> dcacascdau{"dcacascdau", 1.0, "DCA cascade Daughters"};
    Configurable<float> lambdaMassWindow{"lambdaMassWindow", .010, "Distance from Lambda mass (does not apply to KF path)"};
    Configurable<float> maxDaughterEta{"maxDaughterEta", 5.0, "Maximum daughter eta (in abs value)"};
  } cascadeBuilderOpts;

    // V0 building options
  struct : ConfigurableGroup {
    std::string prefix = "MCAssociationOpts";
    Configurable<bool> doMCAssociation{"doMCAssociation", true, "If True, select mothers and daughters based on MC association"};

    Configurable<int> pdgCodeMother{"pdgCodeMother", 22, "PDG Code for V0"};
    Configurable<int> pdgCodeNeg{"pdgCodeNeg", 11, "PDG Code for Negative Track"};
    Configurable<int> pdgCodePos{"pdgCodePos", -11, "PDG Code for Positive Track"};
  
  } MCAssociationOpts;

  // Axis
  // base properties
  ConfigurableAxis axisPt{"axisPt", {VARIABLE_WIDTH, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 17.0f, 19.0f, 21.0f, 23.0f, 25.0f, 30.0f, 35.0f, 40.0f, 50.0f}, "pt axis for analysis"};
  ConfigurableAxis axisPA{"axisPA", {200, 0.0f, 2.0f}, "Pointing angle"};
  ConfigurableAxis axisBDTScore{"axisBDTScore", {200, 0.0f, 1.0f}, "BDT Score"};
  ConfigurableAxis axisDCAz{"axisDCAz", {200, -50.0f, 50.0f}, "DCAz"};
  
  int v0GroupGlobalID = 0;
  void init(InitContext const&)
  {
    histos.add("hDuplicateCount", "hDuplicateCount", kTH1F, {{50, -0.5f, 49.5f}});
    histos.add("hDuplicateCountType7", "hDuplicateCountType7", kTH1F, {{50, -0.5f, 49.5f}});
    histos.add("hDuplicateCountType7allTPConly", "hDuplicateCountType7allTPConly", kTH1F, {{50, -0.5f, 49.5f}});

    histos.add("hPhotonPt", "hPhotonPt", kTH1F, {axisPt});
    histos.add("hPhotonPt_Duplicates", "hPhotonPt_Duplicates", kTH1F, {axisPt});
    histos.add("hPhotonPt_withRecoedMcCollision", "hPhotonPt_withRecoedMcCollision", kTH1F, {axisPt});
    histos.add("hPhotonPt_withCorrectCollisionCopy", "hPhotonPt_withCorrectCollisionCopy", kTH1F, {axisPt});

    histos.add("hPhotonTest_Correct", "hPhotonTest_Correct", kTH1F, {axisPt});
    histos.add("hPhotonTest_Wrong", "hPhotonTest_Wrong", kTH1F, {axisPt});
    histos.add("hPhotonTestPA_Correct", "hPhotonTestPA_Correct", kTH1F, {axisPA});
    histos.add("hPhotonTestPA_Wrong", "hPhotonTestPA_Wrong", kTH1F, {axisPA});
    histos.add("hPhotonTestY_Correct", "hPhotonTestY_Correct", kTH1F, {{200, -2.0f, 2.0f}});
    histos.add("hPhotonTestY_Wrong", "hPhotonTestY_Wrong", kTH1F, {{200, -2.0f, 2.0f}});    
  
    // winner-takes-all criteria
    histos.add("hCorrect_BestPA", "hCorrect_BestPA", kTH1F, {axisPt});
    histos.add("hWrong_BestPA", "hWrong_BestPA", kTH1F, {axisPt});
    histos.add("hCorrect_BestDCADau", "hCorrect_DCADau", kTH1F, {axisPt});
    histos.add("hWrong_BestDCADau", "hWrong_BestDCADau", kTH1F, {axisPt});
    histos.add("hCorrect_BestPAandDCADau", "hCorrect_BestPAandDCADau", kTH1F, {axisPt});
    histos.add("hWrong_BestPAandDCADau", "hWrong_BestPAandDCADau", kTH1F, {axisPt});
    histos.add("hCorrect_BDTScore", "hCorrect_BDTScore", kTH1F, {axisPt});
    histos.add("hWrong_BDTScore", "hWrong_BDTScore", kTH1F, {axisPt});
    
    // Selection criteria 
    auto h3dPAVsPt = histos.add<TH3>("h3dPAVsPt", "h3dPAVsPt", kTH3F, {{2, -0.5f, 1.5f}, axisPA, axisPt});
    h3dPAVsPt->GetXaxis()->SetBinLabel(1, "Wrong Association");
    h3dPAVsPt->GetXaxis()->SetBinLabel(2, "Correct collision");

    histos.add("hMLScore", "hMLScore", kTH1F, {axisBDTScore});
    auto h3dMLScoreVsPt = histos.add<TH3>("h3dMLScoreVsPt", "h3dMLScoreVsPt", kTH3F, {{2, -0.5f, 1.5f}, axisBDTScore, axisPt});
    h3dMLScoreVsPt->GetXaxis()->SetBinLabel(1, "Wrong Association");
    h3dMLScoreVsPt->GetXaxis()->SetBinLabel(2, "Correct collision");
    
    // Setup ccdb   
    ccdb->setURL(ccdbConfigurations.ccdburl);
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();
    ccdb->setFatalWhenNull(false);

    // set V0 parameters in the helper
    straHelper.v0selections.minCrossedRows = v0BuilderOpts.minCrossedRows;
    straHelper.v0selections.dcanegtopv = v0BuilderOpts.dcanegtopv;
    straHelper.v0selections.dcapostopv = v0BuilderOpts.dcapostopv;
    straHelper.v0selections.v0cospa = v0BuilderOpts.v0cospa;
    straHelper.v0selections.dcav0dau = v0BuilderOpts.dcav0dau;
    straHelper.v0selections.v0radius = v0BuilderOpts.v0radius;
    straHelper.v0selections.maxDaughterEta = v0BuilderOpts.maxDaughterEta;

    // set cascade parameters in the helper
    straHelper.cascadeselections.minCrossedRows = cascadeBuilderOpts.minCrossedRows;
    straHelper.cascadeselections.dcabachtopv = cascadeBuilderOpts.dcabachtopv;
    straHelper.cascadeselections.cascradius = cascadeBuilderOpts.cascradius;
    straHelper.cascadeselections.casccospa = cascadeBuilderOpts.casccospa;
    straHelper.cascadeselections.dcacascdau = cascadeBuilderOpts.dcacascdau;
    straHelper.cascadeselections.lambdaMassWindow = cascadeBuilderOpts.lambdaMassWindow;
    straHelper.cascadeselections.maxDaughterEta = cascadeBuilderOpts.maxDaughterEta;
        
    if (PredictV0Association) {
      if (loadModelsFromCCDB) { 
        // Retrieve the model from CCDB
        ccdbApi.init(ccdbUrl);

        /// Fetching model for specific timestamp
        LOG(info) << "Fetching model for timestamp: " << timestampCCDB.value;
        
        bool retrieveSuccess = ccdbApi.retrieveBlob(BDTPathCCDB.value, ".", metadata, timestampCCDB.value, false, BDTLocalPath.value);
        if (retrieveSuccess) {
          deduplication_bdt.initModel(BDTLocalPath.value, enableOptimizations.value);
        } else {
          LOG(fatal) << "Error encountered while fetching/loading the Gamma model from CCDB! Maybe the model doesn't exist yet for this runnumber/timestamp?";
        }
      }
      else{
        deduplication_bdt.initModel(BDTLocalPath.value, enableOptimizations.value);      
      }         
    } 
  }

  template <typename TCollisions>
  bool initCCDB(aod::BCsWithTimestamps const& bcs, TCollisions const& collisions)
  {
    auto bc = collisions.size() ? collisions.begin().template bc_as<aod::BCsWithTimestamps>() : bcs.begin();
    if (!bcs.size()) {
      LOGF(warn, "No BC found, skipping this DF.");
      return false; // signal to skip this DF
    }

    if (mRunNumber == bc.runNumber()) {
      return true;
    }

    auto timestamp = bc.timestamp();
    o2::parameters::GRPMagField* grpmag = 0x0;

    grpmag = ccdb->getForTimeStamp<o2::parameters::GRPMagField>(ccdbConfigurations.grpmagPath, timestamp);
    if (!grpmag) {
      LOG(fatal) << "Got nullptr from CCDB for path " << ccdbConfigurations.grpmagPath << " of object GRPMagField for timestamp " << timestamp;
    }
    o2::base::Propagator::initFieldFromGRP(grpmag);

    // Fetch magnetic field from ccdb for current collision
    auto magneticField = o2::base::Propagator::Instance()->getNominalBz();
    LOG(info) << "Retrieved GRP for timestamp " << timestamp << " with magnetic field of " << magneticField << " kG";

    // Set magnetic field value once known
    straHelper.fitter.setBz(magneticField);

    // acquire LUT for this timestamp
    LOG(info) << "Loading material look-up table for timestamp: " << timestamp;
    lut = o2::base::MatLayerCylSet::rectifyPtrFromFile(ccdb->getForTimeStamp<o2::base::MatLayerCylSet>(ccdbConfigurations.lutPath, timestamp));
    o2::base::Propagator::Instance()->setMatLUT(lut);
    straHelper.lut = lut;

    LOG(info) << "Fully configured for run: " << bc.runNumber();
    // mmark this run as configured
    mRunNumber = bc.runNumber();

    // initialize only if needed, avoid unnecessary CCDB calls
    mVDriftMgr.init(&ccdb->instance());
    mVDriftMgr.update(timestamp);

    return true;
  }
  
  //_______________________________________________________________________
  template <typename TMCParticles>
  int findMotherFromLabels(int const& p1, int const& p2, TMCParticles const& mcparticles)
  {
    // encompasses a simple check for labels existing
    if (p1 < 0 || p2 < 0) {
      return -1;
    }
    auto mcParticle1 = mcparticles.rawIteratorAt(p1);
    auto mcParticle2 = mcparticles.rawIteratorAt(p2);
    return (findMother(mcParticle1, mcParticle2, mcparticles));
  }

  //_______________________________________________________________________
  template <typename TMCParticle1, typename TMCParticle2, typename TMCParticles>
  int findMother(TMCParticle1 const& p1, TMCParticle2 const& p2, TMCParticles const& mcparticles)
  {
    if (p1.globalIndex() == p2.globalIndex())
      return -1;
    if ((p1.pdgCode() != MCAssociationOpts.pdgCodePos || p2.pdgCode() != MCAssociationOpts.pdgCodeNeg) && MCAssociationOpts.doMCAssociation)
      return -1;
    if (!p1.has_mothers() || !p2.has_mothers())
      return -1;

    int motherid1 = p1.mothersIds()[0];
    auto mother1 = mcparticles.iteratorAt(motherid1);
    int mother1_pdg = mother1.pdgCode();
    int motherid2 = p2.mothersIds()[0];

    if (motherid1 != motherid2) 
      return -1;

    if((mother1_pdg != MCAssociationOpts.pdgCodeMother) && MCAssociationOpts.doMCAssociation)
      return -1; // not the expected mother

    return motherid1;
  }

    //__________________________________________
  // Helper structure to save v0 duplicates auxiliary info
  struct V0DuplicateExtra {
    float collX;
    float collY;
    float collZ;
    float collTime;
    float v0PosTrackTime;
    float v0NegTrackTime;
    float v0DauDCAxy;
    float v0DauDCAz;
    float v0DCAToPVxy;
    float v0DCAToPVz;
    float v0MCpT;
    float v0Px;
    float v0Py;
    float v0Pz;
    float v0PhotonMass; 
    float v0PhotonY;
    int v0GroupGlobalID;    
    bool v0IsCorrectlyAssociated;
    bool v0hasCorrectCollisionCopy;
    int v0PDGCode;
    bool isBuildOk;
    bool IsPrimary;
  };

  // Simple function to sort values in a vector
  vector<int> rankSort(const vector<float>& v_temp, bool descending = false) {
    vector<pair<float, size_t>> v_sort(v_temp.size());

    // Pair each value with its original index
    for (size_t i = 0U; i < v_temp.size(); ++i) {
        v_sort[i] = make_pair(v_temp[i], i);
    }

    // Sort by value - ascending: lowest gets rank 1, descending: highest gets rank 1

    if (descending) {
        std::sort(v_sort.begin(), v_sort.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });
    } else {
        std::sort(v_sort.begin(), v_sort.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
    }

    pair<float, size_t> rank_tracker = make_pair(std::numeric_limits<float>::quiet_NaN(), 0);
    vector<int> result(v_temp.size());

    for (size_t i = 0U; i < v_sort.size(); ++i) {
        // Only update rank if value is different from previous
        if (v_sort[i].first != rank_tracker.first) {
            rank_tracker = make_pair(v_sort[i].first, i + 1);  // +1 for 1-based rank
        }
        result[v_sort[i].second] = rank_tracker.second;  // assign rank to original index
    }

    return result;
  }

  //_______________________________________________________________________
  // Process duplicated photons
  void processDuplicates(std::vector<o2::pwglf::v0candidate> v0duplicates, std::vector<V0DuplicateExtra> V0DuplicateExtras, std::vector<o2::pwglf::V0group> V0Grouped, size_t iV0)
  {
    float bestBDTScore = -1;
    float bestPointingAngle = 1000; // non sense value
    float bestDCADaughters = 1e+6;

    bool bestBDTScoreCorrect = false;
    bool bestPointingAngleCorrect = false;
    bool bestDCADaughtersCorrect = false;    

    // Defining context variables
    int NDuplicates = 0;
    
    float AvgPA = 0.0f;    
    float AvgZ = 0.0f;    
    float AvgDCADauxy = 0.0f;
    float AvgDCADauz = 0.0f;
    float AvgV0DCAxy = 0.0f;
    float AvgV0DCAz = 0.0f;
    float MinPA = 999.f;
    float MaxPA = -999.f;
    float MinZ = 999.f;
    float MaxZ = -999.f;
    float MinDCADauxy = 999.f;
    float MaxDCADauxy = -999.f;
    float MinDCADauz = 999.f;
    float MaxDCADauz = -999.f;
    float MinV0DCAxy = 999.f;
    float MaxV0DCAxy = -999.f;
    float MinV0DCAz = 999.f;
    float MaxV0DCAz = -999.f;

    // Containers for ranking
    std::vector<float> paVec(V0Grouped[iV0].collisionIds.size(), 999.f);
    std::vector<float> v0zVec(V0Grouped[iV0].collisionIds.size(), 999.f);
    std::vector<float> DCADauxyVec(V0Grouped[iV0].collisionIds.size(), 999.f);
    std::vector<float> DCADauzVec(V0Grouped[iV0].collisionIds.size(), 999.f);
    std::vector<float> v0DCAxyVec(V0Grouped[iV0].collisionIds.size(), 999.f);
    std::vector<float> v0DCAzVec(V0Grouped[iV0].collisionIds.size(), 999.f);
    
    for (size_t ic = 0; ic < V0Grouped[iV0].collisionIds.size(); ic++) {  
      // --------------------------------------------------------------------------
      // ADDITIONAL SELECTION CRITERIA                
      if (!V0DuplicateExtras[ic].isBuildOk && v0BuilderOpts.selectBuiltOnly) {
        continue; // skip not built V0s
      }

      if (V0DuplicateExtras[ic].v0PhotonMass > v0BuilderOpts.PhotonMaxMass)
        continue; // skip anything that doesn't look like a photon

      if (!V0DuplicateExtras[ic].IsPrimary && v0BuilderOpts.selectPrimariesOnly)
        continue;

      if (TMath::Abs(V0DuplicateExtras[ic].v0PhotonY) > v0BuilderOpts.V0Rapidity) 
        continue;

      // TODO: include optional pT selection
      
      float pa = v0duplicates[ic].pointingAngle;
      float z = std::abs(v0duplicates[ic].position[2]);
      float dcaDauxy = std::abs(V0DuplicateExtras[ic].v0DauDCAxy);
      float dcaDauz = std::abs(V0DuplicateExtras[ic].v0DauDCAz);
      float dcaToPVxy = std::abs(V0DuplicateExtras[ic].v0DCAToPVxy);
      float dcaToPVz = std::abs(V0DuplicateExtras[ic].v0DCAToPVz);
      
      AvgPA += pa;
      AvgZ += z;
      AvgDCADauxy += dcaDauxy;
      AvgDCADauz += dcaDauz;
      AvgV0DCAxy += dcaToPVxy;
      AvgV0DCAz += dcaToPVz;

      if (pa < MinPA) MinPA = pa;
      if (pa > MaxPA) MaxPA = pa;
      if (z < MinZ) MinZ = z;
      if (z > MaxZ) MaxZ = z;
      if (dcaDauxy < MinDCADauxy) MinDCADauxy = dcaDauxy;
      if (dcaDauxy > MaxDCADauxy) MaxDCADauxy = dcaDauxy;
      if (dcaDauz < MinDCADauz) MinDCADauz = dcaDauz;
      if (dcaDauz > MaxDCADauz) MaxDCADauz = dcaDauz;
      if (dcaToPVxy < MinV0DCAxy) MinV0DCAxy = dcaToPVxy;
      if (dcaToPVxy > MaxV0DCAxy) MaxV0DCAxy = dcaToPVxy;
      if (dcaToPVz < MinV0DCAz) MinV0DCAz = dcaToPVz;
      if (dcaToPVz > MaxV0DCAz) MaxV0DCAz = dcaToPVz;
      
      // Filling vector for ranking
      paVec[ic] = pa;
      v0zVec[ic] = z;
      DCADauxyVec[ic] = dcaDauxy;
      DCADauzVec[ic] = dcaDauz;
      v0DCAxyVec[ic] = dcaToPVxy;
      v0DCAzVec[ic] = dcaToPVz;
      
      NDuplicates++;
    }

    // Finalize averages
    if (NDuplicates > 0) {
      AvgPA /= NDuplicates;
      AvgZ /= NDuplicates;
      AvgDCADauxy /= NDuplicates;
      AvgDCADauz /= NDuplicates;
      AvgV0DCAxy /= NDuplicates;
      AvgV0DCAz /= NDuplicates;
    }

    // Get vector of ranks
    std::vector<int> paRanks = rankSort(paVec, false);
    std::vector<int> v0zRanks = rankSort(v0zVec, false);
    std::vector<int> dcaDauxyRanks = rankSort(DCADauxyVec, false);
    std::vector<int> dcaDauzRanks = rankSort(DCADauzVec, false);
    std::vector<int> v0DCAxyRanks = rankSort(v0DCAxyVec, false);
    std::vector<int> v0DCAzRanks = rankSort(v0DCAzVec, false);

    // fill duplicates table
    for (size_t ic = 0; ic < V0Grouped[iV0].collisionIds.size(); ic++) {  
      // --------------------------------------------------------------------------
      // ADDITIONAL SELECTION CRITERIA              
      if (!V0DuplicateExtras[ic].isBuildOk && v0BuilderOpts.selectBuiltOnly) {
        continue; // skip not built V0s
      }
      if (V0DuplicateExtras[ic].v0PhotonMass > v0BuilderOpts.PhotonMaxMass)
        continue; // skip anything that doesn't look like a photon

      if (!V0DuplicateExtras[ic].IsPrimary && v0BuilderOpts.selectPrimariesOnly)
        continue;

      if (TMath::Abs(V0DuplicateExtras[ic].v0PhotonY) > v0BuilderOpts.V0Rapidity) 
        continue;

      // TODO: include optional pT selection

      // --------------------------------------------------------------------------
      // PHOTON DUPLICATES PROPERTIES
      float v0px = V0DuplicateExtras[ic].v0Px;
      float v0py = V0DuplicateExtras[ic].v0Py;
      float v0pz = V0DuplicateExtras[ic].v0Pz;

      float v0Z = std::abs(v0duplicates[ic].position[2]);
      float v0DCADau = std::abs(v0duplicates[ic].daughterDCA);
      float v0DauDCAxy = std::abs(V0DuplicateExtras[ic].v0DauDCAxy);
      float v0DauDCAz = std::abs(V0DuplicateExtras[ic].v0DauDCAz);
      float v0PA = v0duplicates[ic].pointingAngle;
      float v0Radius = RecoDecay::sqrtSumOfSquares(v0duplicates[ic].position[0], v0duplicates[ic].position[1]);
      float v0PosDCAToPV = std::abs(v0duplicates[ic].positiveDCAxy);
      float v0NegDCAToPV = std::abs(v0duplicates[ic].negativeDCAxy);
      float v0DCAToPVxy = std::abs(V0DuplicateExtras[ic].v0DCAToPVxy);
      float v0DCAToPVz = std::abs(V0DuplicateExtras[ic].v0DCAToPVz);
      float v0Phi = RecoDecay::phi(v0px, v0py);
      float collX = V0DuplicateExtras[ic].collX;
      float collY = V0DuplicateExtras[ic].collY;
      float collZ = V0DuplicateExtras[ic].collZ;      

      float v0PhotonMass = V0DuplicateExtras[ic].v0PhotonMass;
      float v0pt = RecoDecay::sqrtSumOfSquares(v0px, v0py);
      float v0mcpt = V0DuplicateExtras[ic].v0MCpT;
      
      float v0Y = V0DuplicateExtras[ic].v0PhotonY;
      float v0Eta = RecoDecay::eta(std::array{v0px, v0py, v0pz});
      
      float v0PosTrackTime = V0DuplicateExtras[ic].v0PosTrackTime;
      float v0NegTrackTime = V0DuplicateExtras[ic].v0NegTrackTime;
      float collTime = V0DuplicateExtras[ic].collTime;
    
      int PARank = paRanks[ic];
      int ZRank = v0zRanks[ic];
      int DCADauxyRank = dcaDauxyRanks[ic];
      int DCADauzRank = dcaDauzRanks[ic];
      int v0DCAxyRank = v0DCAxyRanks[ic];
      int v0DCAzRank = v0DCAzRanks[ic];

      // Auxiliary variables
      int v0GroupID = V0DuplicateExtras[ic].v0GroupGlobalID;
      int PDGCode = V0DuplicateExtras[ic].v0PDGCode;
      bool v0IsCorrectlyAssociated = V0DuplicateExtras[ic].v0IsCorrectlyAssociated;
      
      // --------------------------------------------------------------------------
      // De-duplication strategy tests start here

      // Check for deduplication criteria:
      histos.fill(HIST("hPhotonPt_Duplicates"), v0mcpt);            
      histos.fill(HIST("h3dPAVsPt"), v0IsCorrectlyAssociated, v0PA, v0mcpt);              

      if (v0IsCorrectlyAssociated){
        histos.fill(HIST("hPhotonTest_Correct"), v0mcpt);
        histos.fill(HIST("hPhotonTestPA_Correct"), v0PA);
        histos.fill(HIST("hPhotonTestY_Correct"), v0Y);
      }
      else{
        histos.fill(HIST("hPhotonTest_Wrong"), v0mcpt);
        histos.fill(HIST("hPhotonTestPA_Wrong"), v0PA);
        histos.fill(HIST("hPhotonTestY_Wrong"), v0Y);
      }
      // The winner takes it all criteria
      // Best PA wins
      if (v0PA < bestPointingAngle) {
        bestPointingAngle = v0PA;
        bestPointingAngleCorrect = v0IsCorrectlyAssociated;
      }

      // Best DCADau wins
      if (v0DCADau < bestDCADaughters) {
        bestDCADaughters = v0DCADau;
        bestDCADaughtersCorrect = v0IsCorrectlyAssociated;
      }      
  
      // BDT scores treatment
      float BDTScore = -1.0f;
      if (PredictV0Association) {
        // Define input features for the BDT
        std::vector<float> inputFeatures{v0DCAToPVz, v0PA, v0Z, static_cast<float>(PARank), static_cast<float>(NDuplicates), 
                                         AvgPA, static_cast<float>(ZRank), v0Radius, MaxZ, MinPA, static_cast<float>(v0DCAzRank), v0Eta};

        float* BDTProbability = deduplication_bdt.evalModel(inputFeatures);
        BDTScore = BDTProbability[1]; 
        
        // Fill histograms for deduplication using selections
        histos.fill(HIST("hMLScore"), BDTScore);        
        histos.fill(HIST("h3dMLScoreVsPt"), v0IsCorrectlyAssociated, BDTScore, v0mcpt);        

        // Best BDT Score wins
        if (BDTScore > bestBDTScore) {
          bestBDTScore = BDTScore;
          bestBDTScoreCorrect = v0IsCorrectlyAssociated;
        }
      }
      
      // --------------------------------------------------------------------------
      // Fill table
      if (fillDuplicatesTable) 
        photonDuplicates(v0Z, v0DCADau, v0DauDCAxy, v0DauDCAz, v0PA, v0Radius, v0PosDCAToPV, v0NegDCAToPV, v0DCAToPVxy, v0DCAToPVz, v0Phi,
                       collX, collY, collZ, AvgDCADauxy, AvgDCADauz, AvgPA, AvgZ,
                       MinPA, MaxPA, MinZ, MaxZ, MinDCADauxy, MaxDCADauxy, MinDCADauz, MaxDCADauz,
                       MinV0DCAxy, MaxV0DCAxy, MinV0DCAz, MaxV0DCAz,
                       PARank, ZRank, DCADauxyRank, DCADauzRank, v0DCAxyRank, v0DCAzRank,
                       v0PhotonMass, v0pt, v0px, v0py, v0pz, v0Y, v0Eta, 
                       v0PosTrackTime, v0NegTrackTime, collTime, NDuplicates,
                       v0GroupID, PDGCode, v0IsCorrectlyAssociated);
    }  // end duplicate loop

    // check individual criteria for winner-is-correct
    if (V0DuplicateExtras[0].v0hasCorrectCollisionCopy){ 

      // Deduplication mode 1
      if (bestPointingAngleCorrect) 
        histos.fill(HIST("hCorrect_BestPA"), V0DuplicateExtras[0].v0MCpT);      
      else
        histos.fill(HIST("hWrong_BestPA"), V0DuplicateExtras[0].v0MCpT);      

      // Deduplication mode 2
      if (bestDCADaughtersCorrect) 
        histos.fill(HIST("hCorrect_BestDCADau"), V0DuplicateExtras[0].v0MCpT);      
      else 
        histos.fill(HIST("hWrong_BestDCADau"), V0DuplicateExtras[0].v0MCpT); 
      
      // Deduplication mode 3
      if (bestDCADaughtersCorrect && bestPointingAngleCorrect) 
        histos.fill(HIST("hCorrect_BestPAandDCADau"), V0DuplicateExtras[0].v0MCpT);      
      else 
        histos.fill(HIST("hWrong_BestPAandDCADau"), V0DuplicateExtras[0].v0MCpT); // TODO: declare this    
      
      // Deduplication mode 4
      if (PredictV0Association){
        if (bestBDTScoreCorrect)
          histos.fill(HIST("hCorrect_BDTScore"), V0DuplicateExtras[0].v0MCpT);      
        else
          histos.fill(HIST("hWrong_BDTScore"), V0DuplicateExtras[0].v0MCpT);      
      }    
    }
  }

  void process(soa::Join<aod::Collisions, aod::McCollisionLabels, aod::EvSels> const& collisions, aod::McCollisions const& mcCollisions, aod::V0s const& V0s, LabeledTracksExtra const& tracks, aod::McParticles const& mcParticles, aod::BCsWithTimestamps const& bcs)
  {
    if (!initCCDB(bcs, collisions))
      return;

    std::vector<o2::pwglf::V0group> v0tableGrouped = o2::pwglf::groupDuplicates(V0s);

    // determine map of McCollisions -> Collisions
    std::vector<std::vector<int>> mcCollToColl(mcCollisions.size());
    for (auto const& collision : collisions) {
      if (collision.mcCollisionId() > -1) {
        // useful to determine if collision has been reconstructed afterwards
        mcCollToColl[collision.mcCollisionId()].push_back(collision.globalIndex());
      }
    }

    // simple inspection of grouped duplicates
    for (size_t iV0 = 0; iV0 < v0tableGrouped.size(); iV0++) {
      // base QA histograms
      histos.fill(HIST("hDuplicateCount"), v0tableGrouped[iV0].collisionIds.size());
      if (v0tableGrouped[iV0].v0Type == 7) {
        histos.fill(HIST("hDuplicateCountType7"), v0tableGrouped[iV0].collisionIds.size());
      }

      // Monte Carlo exclusive: process
      auto pTrack = tracks.rawIteratorAt(v0tableGrouped[iV0].posTrackId);
      auto nTrack = tracks.rawIteratorAt(v0tableGrouped[iV0].negTrackId);
      bool pTrackTPCOnly = (pTrack.hasTPC() && !pTrack.hasITS() && !pTrack.hasTRD() && !pTrack.hasTOF());
      bool nTrackTPCOnly = (nTrack.hasTPC() && !nTrack.hasITS() && !nTrack.hasTRD() && !nTrack.hasTOF());
    
      if (v0tableGrouped[iV0].v0Type == 7 && pTrackTPCOnly && nTrackTPCOnly) {
        histos.fill(HIST("hDuplicateCountType7allTPConly"), v0tableGrouped[iV0].collisionIds.size());
      }

      // select on v0type
      if (v0tableGrouped[iV0].v0Type == 0) {
        continue;
      }
      // don't run analysis if no track is TPC only
      if (!pTrackTPCOnly && !nTrackTPCOnly && v0BuilderOpts.skipNonTPCOnly) {
        continue;
      }
      
      int pTrackLabel = pTrack.mcParticleId();
      int nTrackLabel = nTrack.mcParticleId();
      int v0Label = findMotherFromLabels(pTrackLabel, nTrackLabel, mcParticles);
      int correctMcCollision = -1;
      if (v0Label > -1) {
        // this mc particle exists and is a gamma
        auto mcV0 = mcParticles.rawIteratorAt(v0Label);
        correctMcCollision = mcV0.mcCollisionId();

        histos.fill(HIST("hPhotonPt"), mcV0.pt());

        if (mcCollToColl[mcV0.mcCollisionId()].size() > 0) {
          histos.fill(HIST("hPhotonPt_withRecoedMcCollision"), mcV0.pt());
        }

        bool hasCorrectCollisionCopy = false;
        for (size_t ic = 0; ic < v0tableGrouped[iV0].collisionIds.size(); ic++) {
          for (size_t imcc = 0; imcc < mcCollToColl[mcV0.mcCollisionId()].size(); imcc++) {
            if (v0tableGrouped[iV0].collisionIds[ic] == mcCollToColl[mcV0.mcCollisionId()][imcc]) {
              hasCorrectCollisionCopy = true;
            }
          }
        }

        if (hasCorrectCollisionCopy) {
          histos.fill(HIST("hPhotonPt_withCorrectCollisionCopy"), mcV0.pt());
        }

        std::vector<o2::pwglf::v0candidate> v0duplicates; // Vector of v0 candidate duplicates        
        std::vector<V0DuplicateExtra> V0DuplicateExtras; // Vector to store V0 duplicate info
        
        // START OF MAIN DUPLICATE LOOP IS HERE
        for (size_t ic = 0; ic < v0tableGrouped[iV0].collisionIds.size(); ic++) {
          // check if candidate is correctly associated
          bool correctlyAssociated = false;
          for (size_t imcc = 0; imcc < mcCollToColl[correctMcCollision].size(); imcc++) {
            if (v0tableGrouped[iV0].collisionIds[ic] == mcCollToColl[correctMcCollision][imcc]) {
              correctlyAssociated = true;
            }
          }
          
          // actually treat tracks
          auto posTrackPar = getTrackParCov(pTrack);
          auto negTrackPar = getTrackParCov(nTrack);

          auto const& collision = collisions.rawIteratorAt(v0tableGrouped[iV0].collisionIds[ic]);

          // handle TPC-only tracks properly (photon conversions)
          if (v0BuilderOpts.moveTPCOnlyTracks) {            
            if (pTrackTPCOnly) {
              // Nota bene: positive is TPC-only -> this entire V0 merits treatment as photon candidate
              posTrackPar.setPID(o2::track::PID::Electron);
              negTrackPar.setPID(o2::track::PID::Electron);

              if (!mVDriftMgr.moveTPCTrack<aod::BCsWithTimestamps, soa::Join<aod::Collisions, aod::McCollisionLabels, aod::EvSels>>(collision, pTrack, posTrackPar)) {
                return;
              }
            }
            
            if (nTrackTPCOnly) {
              // Nota bene: negative is TPC-only -> this entire V0 merits treatment as photon candidate
              posTrackPar.setPID(o2::track::PID::Electron);
              negTrackPar.setPID(o2::track::PID::Electron);

              if (!mVDriftMgr.moveTPCTrack<aod::BCsWithTimestamps, soa::Join<aod::Collisions, aod::McCollisionLabels, aod::EvSels>>(collision, nTrack, negTrackPar)) {
                return;
              }
            }
          } // end TPC drift treatment

          // process candidate with helper           
          //bool buildOK = straHelper.buildV0Candidate<false>(v0tableGrouped[iV0].collisionIds[ic], collision.posX(), collision.posY(), collision.posZ(), pTrack, nTrack, posTrackPar, negTrackPar, v0tableGrouped[iV0].isCollinearV0, false, true);                              
          bool buildOK = straHelper.buildV0Candidate<true>(v0tableGrouped[iV0].collisionIds[ic], collision.posX(), collision.posY(), collision.posZ(), pTrack, nTrack, posTrackPar, negTrackPar, v0tableGrouped[iV0].isCollinearV0, false, true);                                        
          //bool buildOK = straHelper.buildV0Candidate(v0tableGrouped[iV0].collisionIds[ic], collision.posX(), collision.posY(), collision.posZ(), pTrack, nTrack, posTrackPar, negTrackPar, true, false);                                        
          //bool buildOK = straHelper.buildV0Candidate<false>(v0tableGrouped[iV0].collisionIds[ic], collision.posX(), collision.posY(), collision.posZ(), pTrack, nTrack, posTrackPar, negTrackPar, true, false, true);                              
                    
          float daughterDCAXY = std::hypot(
            straHelper.v0.positivePosition[0] - straHelper.v0.negativePosition[0],
            straHelper.v0.positivePosition[1] - straHelper.v0.negativePosition[1]);
          float daughterDCAZ = std::abs(
            straHelper.v0.positivePosition[2] - straHelper.v0.negativePosition[2]);

          if (!buildOK) {
            daughterDCAXY = daughterDCAZ = 1e+6;
          }
          
          float pxpos = straHelper.v0.positiveMomentum[0]; 
          float pypos = straHelper.v0.positiveMomentum[1];
          float pzpos = straHelper.v0.positiveMomentum[2];
          float pxneg = straHelper.v0.negativeMomentum[0];
          float pyneg = straHelper.v0.negativeMomentum[1];
          float pzneg = straHelper.v0.negativeMomentum[2];

          float v0px = pxpos + pxneg;
          float v0py = pypos + pyneg;
          float v0pz = pzpos + pzneg;

          float v0PhotonMass = RecoDecay::m(std::array{std::array{pxpos, pypos, pzpos}, std::array{pxneg, pyneg, pzneg}}, std::array{o2::constants::physics::MassElectron, o2::constants::physics::MassElectron});
          float v0Y = RecoDecay::y(std::array{v0px, v0py, v0pz}, o2::constants::physics::MassGamma);

          // V0 Duplicates extras
          V0DuplicateExtra v0DuplicateInfo;
          v0DuplicateInfo.collX = collision.posX();
          v0DuplicateInfo.collY = collision.posY();
          v0DuplicateInfo.collZ = collision.posZ();
          v0DuplicateInfo.collTime = collision.collisionTime();
          v0DuplicateInfo.v0PosTrackTime = pTrack.trackTime();
          v0DuplicateInfo.v0NegTrackTime = nTrack.trackTime();
          v0DuplicateInfo.v0DauDCAxy = daughterDCAXY;
          v0DuplicateInfo.v0DauDCAz = daughterDCAZ;
          v0DuplicateInfo.v0DCAToPVxy = straHelper.v0.v0DCAToPVxy;
          v0DuplicateInfo.v0DCAToPVz = straHelper.v0.v0DCAToPVz;
          v0DuplicateInfo.v0MCpT = mcV0.pt();
          v0DuplicateInfo.v0Px = v0px;
          v0DuplicateInfo.v0Py = v0py;
          v0DuplicateInfo.v0Pz = v0pz;
          v0DuplicateInfo.v0PhotonMass = v0PhotonMass;
          v0DuplicateInfo.v0PhotonY = v0Y;
          v0DuplicateInfo.v0GroupGlobalID = v0GroupGlobalID;
          v0DuplicateInfo.v0IsCorrectlyAssociated = correctlyAssociated;
          v0DuplicateInfo.v0hasCorrectCollisionCopy = hasCorrectCollisionCopy;          
          v0DuplicateInfo.v0PDGCode = mcV0.pdgCode();
          v0DuplicateInfo.isBuildOk = buildOK;
          v0DuplicateInfo.IsPrimary = mcV0.isPhysicalPrimary();
  
          // saving duplicates info          
          v0duplicates.push_back(straHelper.v0);
          V0DuplicateExtras.push_back(v0DuplicateInfo);
                
        } // end duplicate loop
                      
        if (fillDuplicatesTable || PredictV0Association) processDuplicates(v0duplicates, V0DuplicateExtras, v0tableGrouped, iV0);
        v0GroupGlobalID++; // Update the global counter, please 

        // printout for inspection
        // TString cosPAString = "";
        // for (size_t iCollisionId = 0; iCollisionId < v0tableGrouped[iV0].collisionIds.size(); iCollisionId++) {
        //   cosPAString.Append(Form("%.5f ", v0duplicates[iCollisionId].pointingAngle));
        // }
        // LOGF(info, "#%i (p,n) = (%i,%i), type %i, point. angles: %s", iV0, v0tableGrouped[iV0].posTrackId, v0tableGrouped[iV0].negTrackId, v0tableGrouped[iV0].v0Type, cosPAString.Data());
      } // end this-is-a-mc-gamma check
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<v0assoqa>(cfgc)};
}
