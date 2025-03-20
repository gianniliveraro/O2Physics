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
//  *+-+*+-+*+-+*+-+*+-+*+-+*
//  v0assocqa task
//  *+-+*+-+*+-+*+-+*+-+*+-+*
//
//  This task loops over a set of V0 indices and
//  checks for v0-to-collision association
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    david.dobrigkeit.chinellato@cern.ch
//    gianni.shigeru.setoue.liveraro@cern.ch

#include <string>
#include <vector>
#include <cmath>
#include <array>
#include <cstdlib>
#include <map>
#include <iterator>
#include <utility>

#include "TRandom3.h"
#include "Framework/runDataProcessing.h"
#include "Framework/RunningWorkflowInfo.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "DCAFitter/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"
#include "PWGLF/DataModel/LFStrangenessMLTables.h"
#include "PWGLF/DataModel/LFParticleIdentification.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsCalibration/MeanVertexObject.h"
#include "TableHelper.h"
#include "Tools/ML/MlResponse.h"
#include "Tools/ML/model.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

// simple checkers
#define bitset(var, nbit) ((var) |= (1 << (nbit)))
#define bitcheck(var, nbit) ((var) & (1 << (nbit)))

// use parameters + cov mat non-propagated, aux info + (extension propagated)
using FullTracksExt = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov>;
using FullTracksExtIU = soa::Join<aod::TracksIU, aod::TracksExtra, aod::TracksCovIU>;
using TracksWithExtra = soa::Join<aod::Tracks, aod::TracksExtra>;

// For dE/dx association in pre-selection
using TracksExtraWithPID = soa::Join<aod::TracksExtra, aod::pidTPCFullEl, aod::pidTPCFullPi, aod::pidTPCFullPr, aod::pidTPCFullHe>;

// For MC and dE/dx association
using TracksExtraWithPIDandLabels = soa::Join<aod::TracksExtra, aod::pidTPCFullEl, aod::pidTPCFullPi, aod::pidTPCFullPr, aod::pidTPCFullHe, aod::McTrackLabels>;

// Pre-selected V0s
using TaggedV0s = soa::Join<aod::V0s, aod::V0Tags>;
using TaggedFindableV0s = soa::Join<aod::FindableV0s, aod::V0Tags>;

// For MC association in pre-selection
using LabeledTracksExtra = soa::Join<aod::TracksExtra, aod::McTrackLabels>;

struct lambdakzeroBuilder {

  void init(InitContext& context)
  {
    auto hCollAssocQA = histos.add<TH2>("hCollAssocQA", "hCollAssocQA", kTH2D, {{6, -0.5f, 5.5f}, {2, -0.5f, 1.5f}});
    hCollAssocQA->GetXaxis()->SetBinLabel(1, "K0");
    hCollAssocQA->GetXaxis()->SetBinLabel(2, "Lambda");
    hCollAssocQA->GetXaxis()->SetBinLabel(3, "AntiLambda");
    hCollAssocQA->GetXaxis()->SetBinLabel(4, "Gamma");
    hCollAssocQA->GetYaxis()->SetBinLabel(1, "Wrong collision");
    hCollAssocQA->GetYaxis()->SetBinLabel(2, "Correct collision");

    auto h2dPtVsCollAssocK0Short = histos.add<TH2>("h2dPtVsCollAssocK0Short", "h2dPtVsCollAssocK0Short", kTH2D, {{100, 0.0f, 10.0f}, {2, -0.5f, 1.5f}});
    auto h2dPtVsCollAssocLambda = histos.add<TH2>("h2dPtVsCollAssocLambda", "h2dPtVsCollAssocLambda", kTH2D, {{100, 0.0f, 10.0f}, {2, -0.5f, 1.5f}});
    auto h2dPtVsCollAssocAntiLambda = histos.add<TH2>("h2dPtVsCollAssocAntiLambda", "h2dPtVsCollAssocAntiLambda", kTH2D, {{100, 0.0f, 10.0f}, {2, -0.5f, 1.5f}});
    auto h2dPtVsCollAssocGamma = histos.add<TH2>("h2dPtVsCollAssocGamma", "h2dPtVsCollAssocGamma", kTH2D, {{100, 0.0f, 10.0f}, {2, -0.5f, 1.5f}});

    h2dPtVsCollAssocK0Short->GetYaxis()->SetBinLabel(1, "Wrong collision");
    h2dPtVsCollAssocK0Short->GetYaxis()->SetBinLabel(2, "Correct collision");
    h2dPtVsCollAssocLambda->GetYaxis()->SetBinLabel(1, "Wrong collision");
    h2dPtVsCollAssocLambda->GetYaxis()->SetBinLabel(2, "Correct collision");
    h2dPtVsCollAssocAntiLambda->GetYaxis()->SetBinLabel(1, "Wrong collision");
    h2dPtVsCollAssocAntiLambda->GetYaxis()->SetBinLabel(2, "Correct collision");
    h2dPtVsCollAssocGamma->GetYaxis()->SetBinLabel(1, "Wrong collision");
    h2dPtVsCollAssocGamma->GetYaxis()->SetBinLabel(2, "Correct collision");
  }

  //_______________________________________________________________________
  template <typename TMCParticle1, typename TMCParticle2, typename TMCParticles>
  int FindCommonMotherFrom2Prongs(TMCParticle1 const& p1, TMCParticle2 const& p2, const int expected_pdg1, const int expected_pdg2, const int expected_mother_pdg, TMCParticles const& mcparticles)
  {
    if (p1.globalIndex() == p2.globalIndex())
      return -1; // mc particle p1 and p2 is identical. reject.

    if (p1.pdgCode() != expected_pdg1)
      return -1;
    if (p2.pdgCode() != expected_pdg2)
      return -1;

    if (!p1.has_mothers())
      return -1;
    if (!p2.has_mothers())
      return -1;

    // LOGF(info,"original motherid1 = %d , motherid2 = %d", p1.mothersIds()[0], p2.mothersIds()[0]);

    int motherid1 = p1.mothersIds()[0];
    auto mother1 = mcparticles.iteratorAt(motherid1);
    int mother1_pdg = mother1.pdgCode();

    int motherid2 = p2.mothersIds()[0];
    auto mother2 = mcparticles.iteratorAt(motherid2);
    int mother2_pdg = mother2.pdgCode();

    // LOGF(info,"motherid1 = %d , motherid2 = %d", motherid1, motherid2);

    if (motherid1 != motherid2)
      return -1;
    if (mother1_pdg != mother2_pdg)
      return -1;
    if (mother1_pdg != expected_mother_pdg)
      return -1;
    return motherid1;
  }

  //*+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*
  /// function to check PDG association
  template <class TTrackTo, typename TV0Object, typename TMCParticles>
  void checkPDG(TV0Object const& lV0Candidate, uint32_t& maskElement, int mcCollisionId, TMCParticles mcparticles)
  {
    int lPDG = -1;
    int correctMcCollisionIndex = -1;
    int photonid = -1;
    float mcpt = -1.0;
    bool physicalPrimary = false;
    auto lNegTrack = lV0Candidate.template negTrack_as<TTrackTo>();
    auto lPosTrack = lV0Candidate.template posTrack_as<TTrackTo>();

    // Association check
    // There might be smarter ways of doing this in the future
    if (lNegTrack.has_mcParticle() && lPosTrack.has_mcParticle()) {
      auto lMCNegTrack = lNegTrack.template mcParticle_as<aod::McParticles>();
      auto lMCPosTrack = lPosTrack.template mcParticle_as<aod::McParticles>();
      // Test for photons!
      photonid = FindCommonMotherFrom2Prongs(lMCPosTrack, lMCNegTrack, -11, 11, 22, mcparticles);
      if (lMCNegTrack.has_mothers() && lMCPosTrack.has_mothers()) {
        
        for (auto& lNegMother : lMCNegTrack.template mothers_as<aod::McParticles>()) {
          for (auto& lPosMother : lMCPosTrack.template mothers_as<aod::McParticles>()) {
            if (lNegMother.globalIndex() == lPosMother.globalIndex() && (!dIfMCselectPhysicalPrimary || lNegMother.isPhysicalPrimary())) {
              lPDG = lNegMother.pdgCode();
              correctMcCollisionIndex = lNegMother.mcCollisionId();
              physicalPrimary = lNegMother.isPhysicalPrimary();
              mcpt = lNegMother.pt();

              // additionally check PDG of the mother particle if requested
              if (dIfMCselectV0MotherPDG != 0) {
                lPDG = 0; // this is not the species you're looking for
                if (lNegMother.has_mothers()) {
                  for (auto& lNegGrandMother : lNegMother.template mothers_as<aod::McParticles>()) {
                    if (lNegGrandMother.pdgCode() == dIfMCselectV0MotherPDG)
                      lPDG = lNegMother.pdgCode();
                  }
                }
              }
              // end extra PDG of mother check
            }
          }
        }
      }
      
    } // end association check

    bool collisionAssociationOK = false;
    if (correctMcCollisionIndex > -1 && correctMcCollisionIndex == mcCollisionId) {
      collisionAssociationOK = true;
    }

    if (lPDG == 310) {
      if (qaCollisionAssociation) {
        histos.fill(HIST("hCollAssocQA"), 0.0f, collisionAssociationOK);
        histos.fill(HIST("h2dPtVsCollAssocK0Short"), mcpt, collisionAssociationOK);
      }
    }
    if (lPDG == 3122) {
      if (qaCollisionAssociation) {
        histos.fill(HIST("hCollAssocQA"), 1.0f, collisionAssociationOK);
        histos.fill(HIST("h2dPtVsCollAssocLambda"), mcpt, collisionAssociationOK);
      }
    }
    if (lPDG == -3122) {
      if (qaCollisionAssociation) {
        histos.fill(HIST("hCollAssocQA"), 2.0f, collisionAssociationOK);
        histos.fill(HIST("h2dPtVsCollAssocAntiLambda"), mcpt, collisionAssociationOK);
      }
    }

    //if (lPDG == 22) { // Small mod to check photon assoc
    if (photonid > 0) { // Small mod to check photon assoc
      if (qaCollisionAssociation) {
        histos.fill(HIST("hCollAssocQA"), 3.0f, collisionAssociationOK);
        histos.fill(HIST("h2dPtVsCollAssocGamma"), mcpt, collisionAssociationOK);
      }
    }
  }

  void processBuildMCAssociated(soa::Join<aod::Collisions, aod::McCollisionLabels> const& /*collisions*/, aod::V0s const& v0table, LabeledTracksExtra const&, aod::McParticles const& particlesMC)
  {
    for (auto const& v0 : v0table) {
      auto collision = v0.collision_as<soa::Join<aod::Collisions, aod::McCollisionLabels>>();
      checkPDG<LabeledTracksExtra>(v0, selectionMask[v0.globalIndex()], collision.mcCollisionId(), particlesMC);
      checkTrackQuality<LabeledTracksExtra>(v0, selectionMask[v0.globalIndex()], true);
    }
  }
  PROCESS_SWITCH(v0assoqa, processBuildMCAssociated, "Process Run 3 data", true);
};
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<v0assoqa>(cfgc)};
}